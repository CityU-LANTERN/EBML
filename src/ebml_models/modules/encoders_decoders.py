import torch
import torch.nn as nn
from torch.nn import Parameter
from collections import OrderedDict
from abc import abstractmethod
import numpy as np
import typing


def mean_pooling(x):
    return torch.mean(x, dim=0, keepdim=True)

class GaussianDecoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.hid_dim = configs.hid_dim
        self.latent_dim = configs.latent_dim
        self.input_dim = 1
        self.mlp = nn.Sequential(
            nn.Linear(1 + self.latent_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, 2)
        )

    def forward(self, x, y, phi):
        '''
        CNP classifier
        '''
        num_phi_samples = phi.shape[0]
        y = y.unsqueeze(0).expand(num_phi_samples, -1, -1, -1)
        x = x.unsqueeze(0).expand(num_phi_samples, -1, -1, -1)
        z = phi.unsqueeze(-2).expand(-1, -1, x.shape[-2], -1)
        xz = torch.cat([x, z], dim=-1)
        e = self.mlp(xz)
        # compute gaussian - log-likelihood @ y
        minus_log_prob = 0.5 * ((y[:, :, :, 0] - e[:, :, :, 0]) / e[:, :, :, 1].exp()).square() + e[:, :, :, 1]
        return minus_log_prob


class EncoderClassification(nn.Module):
    '''
    Compute task-specific context distribution per class, phi_i ~ N(encoder_mean(x,y),encoder_var(x,y))
    for each task in a input task batch of shape: (num_tasks,num_samples_per_task,sample_dim)
    '''

    def __init__(self, configs):
        super().__init__()
        self.latent_dim = configs.model.latent_dim
        self.encode_net1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.encode_net2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.task_mu = nn.Sequential(
            nn.Linear(128, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.task_logsigma = nn.Sequential(
            nn.Linear(128, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

    def forward(self, s_features, s_prototypes):
        # s_feature shape (batch_size, num_ways, num_shot, feature_dim)
        x = s_prototypes
        x = self.encode_net1(x).mean([1])  # pooling along dim ways and shot
        # x = self.encode_net2(x).mean([1])
        task_mean = self.task_mu(x)
        task_sigma = nn.functional.softplus(self.task_logsigma(x)) + 0.1
        # ouput shape:  batch_size,latent_dim
        return task_mean, task_sigma


class TaskEncoder(nn.Module):
    """
    Simple set encoder, implementing the DeepSets approach. Used for modeling permutation invariant representations
    on sets (mainly for extracting task-level representations from context sets).
    """
    def __init__(self,configs):
        super(TaskEncoder, self).__init__()
        self.pre_pooling_fn = SimplePrePoolNet()
        self.latent_dim = configs.model.latent_dim
        # self.post_pooling_fn = nn.Sequential(
        #     nn.Linear(self.pre_pooling_fn.output_size, self.latent_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.latent_dim, self.latent_dim * 2)
        # )
        self.device = configs.device

    def forward(self, x, y):
        """
        Forward pass through DeepSet SetEncoder. Implements the following computation:
        g(X) = rho ( mean ( phi(x) ) )
        Where X = (x0, ... xN) is a set of elements x in X (in our case, images from a context set)
        and the mean is a pooling operation over elements in the set.

        :param x: (torch.tensor) Set of elements X (e.g., for images has shape batch x C x H x W ).
        :return: (torch.tensor) Representation of the set, single vector in Rk.
        """
        x = self.pre_pooling_fn(x).mean(dim=0, keepdim=False)
        # phi_mu, phi_logsigma = self.post_pooling_fn(x).split(self.latent_dim, dim=-1)
        return x, torch.ones_like(x).detach().exp()


class SimplePrePoolNet(nn.Module):
    """
    Simple prepooling network for images. Implements the phi mapping in DeepSets networks. In this work we use a
    multi-layer convolutional network similar to that in https://openreview.net/pdf?id=rJY0-Kcll.
    """

    def __init__(self):
        super(SimplePrePoolNet, self).__init__()
        self.layer1 = self._make_conv2d_layer(3, 64)
        self.layer2 = self._make_conv2d_layer(64, 64)
        self.layer3 = self._make_conv2d_layer(64, 64)
        self.layer4 = self._make_conv2d_layer(64, 64)
        self.layer5 = self._make_conv2d_layer(64, 64)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    @staticmethod
    def _make_conv2d_layer(in_maps, out_maps):
        return nn.Sequential(
            nn.Conv2d(in_maps, out_maps, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_maps,track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    @property
    def output_size(self):
        return 64


class MahaClassifier(nn.Module):
    def __init__(self,configs):
        super(MahaClassifier, self).__init__()
        self.device = configs.device
        self.class_representations = OrderedDict()  # Dictionary mapping class label (integer) to encoded representation
        self.class_precision_matrices = OrderedDict()  # Dictionary mapping class label (integer) to regularized precision matrices estimated

    def forward(self, support_features, support_labels, query_features):
        '''
        :param query_features: adapted query image features, shape : num_phi_samples, num_query, feat_dim
        :return: mean prediction logits, shape : num_query, num_class
        '''
        self._build_class_reps_and_covariance_estimates(support_features, support_labels)
        # get the class means and covariance estimates in tensor form
        class_means = torch.stack(list(self.class_representations.values()), dim=1)
        class_precision_matrices = torch.stack(list(self.class_precision_matrices.values()), dim=1)
        self.class_representations.clear()
        self.class_precision_matrices.clear()
        # grabbing the number of classes and query examples for easier use later in the function
        number_of_classes = class_means.size(1)
        number_of_targets = query_features.size(1)
        # shape: (num_phi,num_targets,num_class,1,feat_dim)
        repeated_target = query_features[:, :, None, None, :].expand(-1, -1, number_of_classes, 1, -1)
        repeated_class_means = class_means[:, None, :, None, :].expand(-1, number_of_targets, -1, 1, -1)
        assert repeated_class_means.shape == repeated_target.shape, f'repeated_class_means {repeated_class_means.shape} should match repeated_target {repeated_target.shape}'
        repeated_difference = (repeated_class_means - repeated_target)
        # shape: (num_phi,1,num_class,feat_dim,feat_dim)
        class_precision_matrices = class_precision_matrices.unsqueeze(dim=1)
        # broadcastable matrix mul : output shape: (num_phi,num_targets,num_class,1,feat_dim)
        first_half = torch.matmul(repeated_difference, class_precision_matrices)
        sample_logits = torch.mul(first_half, repeated_difference).sum(dim=(-1, -2)) * -1
        # shape (num_targets,num_class)
        # mean_pred_prob = nn.functional.softmax(sample_logits,dim=-1).mean(0)
        mean_logits = sample_logits.mean(0)
        # clear all dictionaries
        return mean_logits, class_means

    def _build_class_reps_and_covariance_estimates(self, adapted_features, context_labels):
        """
        Construct and return class level representations and class covariance estimattes for each class in task.
        :param adapted_features: shape (num_phi_samples, task_size, feature_dim)
        :param context_labels: (torch.tensor) Label for each image in the context set.
        :return: (void) Updates the internal class representation and class covariance estimates dictionary.
        """
        task_covariance_estimate = self._estimate_cov(adapted_features)
        for c in torch.unique(context_labels):
            # filter out feature vectors which have class c
            class_features = torch.index_select(adapted_features, 1, self._extract_class_indices(context_labels, c))
            # mean pooling examples to form class means
            class_rep = class_features.mean(dim=1, keepdim=False)
            # updating the class representations dictionary with the mean pooled representation
            self.class_representations[c.item()] = class_rep  # (num_phi_samples,1,rep_dim)
            """
            Calculating the mixing ratio lambda_k_tau for regularizing the class level estimate with the task level estimate."
            Then using this ratio, to mix the two estimate; further regularizing with the identity matrix to assure invertability, and then
            inverting the resulting matrix, to obtain the regularized precision matrix. This tensor is then saved in the corresponding
            dictionary for use later in infering of the query features points.
            """
            lambda_k_tau = (class_features.size(1) / (class_features.size(1) + 1))
            covMatrix = (lambda_k_tau * self._estimate_cov(class_features)) + (
                    (1 - lambda_k_tau) * task_covariance_estimate) \
                        + torch.eye(class_features.size(-1), class_features.size(-1),device=self.device)
            # if covMatrix.abs().sum()<1e-6:
            #    covMatrix += 1e-6*torch.eye(class_features.size(1), class_features.size(1)).cuda(0)
            # self.class_precision_matrices[c.item()] = torch.inverse(covMatrix+1e-6)
            self.class_precision_matrices[c.item()] = torch.linalg.inv(covMatrix)

    def _estimate_cov(self, examples):
        '''
        :param examples: shape : (num_phi_samples, sample_cardinality, sample_dim)
        :return: covariance matrix estimation for each phi observation, shape (num_phi_samples, sample_cardinality x sample_cardinality)
        '''
        assert examples.dim()==3
        # if n_samples == 1
        if examples.size(1)!=1:
            examples = examples.transpose(1,2) # n_phi_samples, sample_dim, n_samples
        # else use the sample_dim to compute the feature dim
        factor = 1.0 / (examples.size(-1) - 1)
        examples = examples - torch.mean(examples, dim=-1, keepdim=True)
        examples_t = examples.transpose(1, 2)
        cov_matrix = factor * examples.matmul(examples_t)  # sample_dim
        assert cov_matrix.shape[1] == cov_matrix.shape[2], f'cov matrix has invalid shape : {cov_matrix.shape}'
        return cov_matrix

    @staticmethod
    def _extract_class_indices(labels, which_class):
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


class MahaClassifierOriginal(nn.Module):
    def __init__(self, configs):
        super(MahaClassifierOriginal, self).__init__()
        print(f'{"":=^50}')
        print(f'{f"Creating Maha Classifier":=^50}')
        print(f'{"":=^50}\n')
        self.device = configs.device
        self.class_representations = OrderedDict()  # Dictionary mapping class label (integer) to encoded representation
        self.class_precision_matrices = OrderedDict()  # Dictionary mapping class label (integer) to regularized
        # precision matrices estimated

    def forward(self, context_features, context_labels, target_features):
        self._build_class_reps_and_covariance_estimates(context_features, context_labels)
        class_means = torch.stack(list(self.class_representations.values())).squeeze(1)
        class_precision_matrices = torch.stack(list(self.class_precision_matrices.values()))

        # grabbing the number of classes and query examples for easier use later in the function
        number_of_classes = class_means.size(0)
        number_of_targets = target_features.size(0)

        """
        Calculating the Mahalanobis distance between query examples and the class means
        including the class precision estimates in the calculations, reshaping the distances
        and multiplying by -1 to produce the sample logits
        """
        repeated_target = target_features.repeat(1, number_of_classes).view(-1, class_means.size(1))
        repeated_class_means = class_means.repeat(number_of_targets, 1)
        repeated_difference = (repeated_class_means - repeated_target)
        repeated_difference = repeated_difference.view(number_of_targets, number_of_classes,
                                                       repeated_difference.size(1)).permute(1, 0, 2)
        first_half = torch.matmul(repeated_difference, class_precision_matrices)
        sample_logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1, 0) * -1

        # clear all dictionaries
        self.class_representations.clear()
        self.class_precision_matrices.clear()

        return sample_logits, class_means

    def _build_class_reps_and_covariance_estimates(self, context_features, context_labels):
        """
        Construct and return class level representations and class covariance estimattes for each class in task.
        :param context_features: (torch.tensor) Adapted feature representation for each image in the context set.
        :param context_labels: (torch.tensor) Label for each image in the context set.
        :return: (void) Updates the internal class representation and class covariance estimates dictionary.
        """
        """
        Calculating a task level covariance estimate using the provided function.
        """
        task_covariance_estimate = self.estimate_cov(context_features)
        for c in torch.unique(context_labels):
            # filter out feature vectors which have class c
            class_features = torch.index_select(context_features, 0, self._extract_class_indices(context_labels, c))
            # mean pooling examples to form class means
            class_rep = mean_pooling(class_features)
            # updating the class representations dictionary with the mean pooled representation
            self.class_representations[c.item()] = class_rep
            """
            Calculating the mixing ratio lambda_k_tau for regularizing the class level estimate with the task level estimate."
            Then using this ratio, to mix the two estimate; further regularizing with the identity matrix to assure invertability, and then
            inverting the resulting matrix, to obtain the regularized precision matrix. This tensor is then saved in the corresponding
            dictionary for use later in infering of the query data points.
            """
            lambda_k_tau = (class_features.size(0) / (class_features.size(0) + 1))
            try:
                self.class_precision_matrices[c.item()] = torch.inverse((lambda_k_tau * self.estimate_cov(class_features)) + ((1 - lambda_k_tau) * task_covariance_estimate) \
                        + torch.eye(class_features.size(1), class_features.size(1)).to(self.device))
            except torch._C._LinAlgError:
                self.class_precision_matrices[c.item()] = torch.eye(class_features.size(1), class_features.size(1)).to(self.device)

    def estimate_cov(self, examples, rowvar=False, inplace=False):
        """
        Function based on the suggested implementation of Modar Tensai
        and his answer as noted in: https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/5

        Estimate a covariance matrix given data.

        Covariance indicates the level to which two variables vary together.
        If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
        then the covariance matrix element `C_{ij}` is the covariance of
        `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

        Args:
            examples: A 1-D or 2-D array containing multiple variables and observations.
                Each row of `m` represents a variable, and each column a single
                observation of all those variables.
            rowvar: If `rowvar` is True, then each row represents a
                variable, with observations in the columns. Otherwise, the
                relationship is transposed: each column represents a variable,
                while the rows contain observations.

        Returns:
            The covariance matrix of the variables.
        """
        if examples.dim() > 2:
            raise ValueError('m has more than 2 dimensions')
        if examples.dim() < 2:
            examples = examples.view(1, -1)
        if not rowvar and examples.size(0) != 1:
            examples = examples.t()
        factor = 1.0 / (examples.size(1) - 1)
        if inplace:
            examples -= torch.mean(examples, dim=1, keepdim=True)
        else:
            examples = examples - torch.mean(examples, dim=1, keepdim=True)
        examples_t = examples.t()
        return factor * examples.matmul(examples_t).squeeze()

    @staticmethod
    def _extract_class_indices(labels, which_class):
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


class ProtoNetClassifier:

    def __init__(self):
        pass

    def __call__(self, features, class_means, phi):
        assert features.shape == class_means.shape, f'repeated_class_means ' \
                                                                    f'{class_means.shape} should match ' \
                                                                    f'repeated_target {features.shape}'
        return (features-class_means).square().sum(-1).sqrt()


class MetricClassifier(nn.Module):
    def __init__(self, configs):
        super(MetricClassifier, self).__init__()
        print(f'{"":=^50}')
        print(f'Creating {configs.model.decoder.upper()} Classifier')
        print(f'{"":=^50}\n')
        self.distance = configs.model.decoder
        self.device = configs.device
        self.class_representations = OrderedDict()  # Dictionary mapping class label (integer) to encoded representation
        # self.class_precision_matrices = OrderedDict()  # Dictionary mapping class label (integer) to regularized
        # precision matrices estimated

    def forward(self, context_features, context_labels, target_features):
        self._build_class_reps_and_covariance_estimates(context_features, context_labels)
        class_means = torch.stack(list(self.class_representations.values())).squeeze(1)
        # class_precision_matrices = torch.stack(list(self.class_precision_matrices.values()))
        # grabbing the number of classes and query examples for easier use later in the function
        # number_of_classes = class_means.size(0)
        # number_of_targets = target_features.size(0)
        assert class_means.dim() == target_features.dim()
        # raise Exception(class_means.shape, target_features.shape)
        if self.distance == 'cosine':
            repeated_class_means = class_means.unsqueeze(0)
            repeated_targets = target_features.unsqueeze(1)
            sample_logits = nn.functional.cosine_similarity(repeated_targets,repeated_class_means,dim=-1, eps=1e-30) * 50
        elif self.distance =='l2':
            repeated_class_means = class_means.unsqueeze(0)
            repeated_targets = target_features.unsqueeze(0)
            sample_logits = -torch.cdist(repeated_targets,repeated_class_means).square().squeeze(0)
        else:
            raise Exception
        # clear all dictionaries
        self.class_representations.clear()
        # self.class_precision_matrices.clear()

        return sample_logits, class_means

    def _build_class_reps_and_covariance_estimates(self, context_features, context_labels):
        """
        Construct and return class level representations and class covariance estimattes for each class in task.
        :param context_features: (torch.tensor) Adapted feature representation for each image in the context set.
        :param context_labels: (torch.tensor) Label for each image in the context set.
        :return: (void) Updates the internal class representation and class covariance estimates dictionary.
        """

        """
        Calculating a task level covariance estimate using the provided function.
        """
        # task_covariance_estimate = self.estimate_cov(context_features)
        for c in torch.unique(context_labels):
            # filter out feature vectors which have class c
            class_features = torch.index_select(context_features, 0, self._extract_class_indices(context_labels, c))
            # mean pooling examples to form class means
            class_rep = mean_pooling(class_features)
            # updating the class representations dictionary with the mean pooled representation
            self.class_representations[c.item()] = class_rep
            # """
            # Calculating the mixing ratio lambda_k_tau for regularizing the class level estimate with the task level
            # estimate."
            # Then using this ratio, to mix the two estimate; further regularizing with the identity matrix to assure
            # invertability, and then
            # inverting the resulting matrix, to obtain the regularized precision matrix. This tensor is then saved in
            # the corresponding
            # dictionary for use later in infering of the query data points.
            # """
            # lambda_k_tau = (class_features.size(0) / (class_features.size(0) + 1))
            # self.class_precision_matrices[c.item()] = torch.inverse(
            #     (lambda_k_tau * self.estimate_cov(class_features)) + ((1 - lambda_k_tau) * task_covariance_estimate) \
            #     + torch.eye(class_features.size(1), class_features.size(1)).to(self.device))

    @staticmethod
    def _extract_class_indices(labels, which_class):
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


class DecoderBase(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.sn = configs.model.decoder_sn
        feat_dim = 512
        sn = nn.utils.spectral_norm if self.sn else lambda x: x
        self.e_net = nn.Sequential(
            sn(nn.Linear(feat_dim * 2 + configs.model.latent_dim, 512)), nn.SiLU(),
            sn(nn.Linear(512, 512)), nn.SiLU(),
            sn(nn.Linear(512, 256)), nn.SiLU(),
            sn(nn.Linear(256, 1))
        )
        self.e_net.apply(self.init_weights)
        self.class_representations=OrderedDict()
        self.class_std_vectors=OrderedDict()
        self.class_precision_matrices = OrderedDict()

    @abstractmethod
    def forward(self, features, class_mean, phi, out='all'):
        raise NotImplementedError

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            # torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.00)

    def get_class_reps(self, adapted_features, context_labels):
        """
        Construct and return class level representations and class covariance estimattes for each class in task.
        :param adapted_features: shape (num_phi_samples, task_size, feature_dim)
        :param context_labels: (torch.tensor) Label for each image in the context set.
        :return: (void) Updates the internal class representation and class covariance estimates dictionary.
        """
        class_representations = OrderedDict()
        for c in torch.unique(context_labels):
            # filter out feature vectors which have class c
            class_features = torch.index_select(adapted_features, 1, self._extract_class_indices(context_labels, c))
            # mean pooling examples to form class means
            class_rep = class_features.mean(dim=1, keepdim=False)
            # updating the class representations dictionary with the mean pooled representation
            class_representations[c.item()] = class_rep  # (num_phi_samples,1,rep_dim)
        return torch.stack(list(class_representations.values()), dim=1)

    def get_class_std_vector(self, adapted_features,context_labels):
        class_representations = OrderedDict()
        for c in torch.unique(context_labels):
            # filter out feature vectors which have class c
            class_features = torch.index_select(adapted_features, 1, self._extract_class_indices(context_labels, c))
            # mean pooling examples to form class means
            class_rep = class_features.std(dim=1, keepdim=False)
            # updating the class representations dictionary with the mean pooled representation
            class_representations[c.item()] = class_rep  # (num_phi_samples,1,rep_dim)
        return torch.stack(list(class_representations.values()), dim=1)

    def adapt_features_prototypes(self, context_features, context_labels, phi):
        prototypes = self.get_class_reps(context_features, context_labels)
        return context_features, prototypes

    @staticmethod
    def _extract_class_indices(labels, which_class):
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


class GenerativeClassifier(DecoderBase):
    def __init__(self, configs):
        super(GenerativeClassifier, self).__init__(configs)
        if configs.model.base_energy =='l2':
            self.base_energy_f = ProtoNetClassifier()
        elif configs.model.base_energy =='cosine':
            self.base_energy_f = MetricClassifier()
        else:
            raise AttributeError
        self.base_energy_weight = configs.model.base_energy_weight

    def forward(self, features, class_mean, phi, out='all',combination=None):
        """
        :param adapted_features: num_samples, num_targets, feat_dim
        :param class_mean: num_samples, num_targets, feat_dim
        :param phi: num_samples, latent_dim
        :return:
        """
        if out== 'e_xy_all' or out== 'e_x':
            num_classes = class_mean.shape[1]
            n_targets = features.shape[1]
            features = features[:, :, None, :].expand(-1, -1, num_classes, -1)
            class_mean = class_mean[:, None, :, :].expand(-1, n_targets, -1, -1)
            phi = phi[:, None, None, :].expand(-1, n_targets, num_classes, -1)
            e_xy = self.e_net(torch.cat([features, class_mean, phi], dim=-1)).squeeze(-1)
        elif out== 'e_xy':
            assert class_mean.shape == features.shape, \
                f'Evaluating E at every pair of xy require matching shape of features and prototypes. ' \
                f'But received features : {features.shape} and prototypes : {class_mean.shape}'
            phi = phi.unsqueeze(-2).expand(-1, features.shape[-2], -1)
            e_xy = self.e_net(torch.cat([features, class_mean, phi], dim=-1)).squeeze(-1)
        else:
            raise AttributeError

        if self.base_energy_f is not None:
            e_xy_base = self.base_energy_f(features, class_mean, phi)
        else:
            e_xy_base = torch.zeros_like(e_xy)

        if combination =='all':
            assert e_xy.shape == e_xy_base.shape, f'Addition between base and NN energy function requires matching tensor shapes. ' \
                f'But received NN : {e_xy.shape} and base : {e_xy_base.shape}'
            e_xy = e_xy + e_xy_base * self.base_energy_weight
        elif combination =='base_only':
            e_xy = e_xy_base * self.base_energy_weight
        elif combination =='ebm_only':
            pass
        else:
            raise KeyError

        if out == 'e_x':
            return -1. * (-e_xy).logsumexp(dim=-1)
        else:
            return e_xy


class DecoderEBM(DecoderBase):
    def __init__(self,configs):
        super(DecoderEBM, self).__init__(configs)

    def forward(self, features, class_mean, phi):
        """
       :param features: num_samples, num_targets, feat_dim
       :param class_mean: num_samples, num_classes, feat_dim
       :param phi: num_samples, latent_dim
       :return:
       """
        num_classes = class_mean.shape[1]
        n_targets = features.shape[1]
        repeated_target = features[:, :, None, :].expand(-1, -1, num_classes, -1)
        repeated_class_means = class_mean[:, None, :, :].expand(-1, n_targets, -1, -1)
        repeated_phi = phi[:,None,None,:].expand(-1, n_targets, num_classes, -1)
        # sample energy
        e_xy = self.e_net(torch.cat([repeated_target, repeated_class_means, repeated_phi], dim=-1)).squeeze(-1)
        return e_xy


class PriorEBM(nn.Module):
    def __init__(self, configs):
        super(PriorEBM, self).__init__()
        self.sn = configs.model.prior_sn
        sn = nn.utils.spectral_norm if self.sn else lambda x: x
        # self.energy_net = nn.Sequential(sn(nn.Linear(configs.model.latent_dim, 256)),
        #                                 nn.ReLU(),
        #                                 sn(nn.Linear(256, 256)),
        #                                 nn.ReLU(),
        #                                 sn(nn.Linear(256, 256)),
        #                                 nn.ReLU(),
        #                                 sn(nn.Linear(256, 1)))

        # self.energy_net = nn.Sequential(sn(nn.Linear(configs.model.latent_dim, 128)),
        #                                 nn.ReLU(),
        #                                 sn(nn.Linear(128, 128)),
        #                                 nn.ReLU(),
        #                                 sn(nn.Linear(128, 64)),
        #                                 nn.ReLU(),
        #                                 sn(nn.Linear(64, 1)))

        self.energy_net = nn.Sequential(sn(nn.Linear(configs.model.latent_dim, 512)),
                                        nn.SiLU(),
                                        sn(nn.Linear(512, 512)),
                                        nn.SiLU(),
                                        sn(nn.Linear(512, 256)),
                                        nn.SiLU(),
                                        sn(nn.Linear(256, 1)))



    def forward(self, task_latent):
        return self.energy_net(task_latent)


class MAHDetector(MahaClassifierOriginal):
    def __init__(self,configs, x,y, tied_covariance=True):
        super(MAHDetector, self).__init__(configs)
        with torch.no_grad():
            self._build_class_reps_and_covariance_estimates(x, y, tied_covariance)

    def forward(self,target_features):
        class_means = torch.stack(list(self.class_representations.values())).squeeze(1)
        class_precision_matrices = torch.stack(list(self.class_precision_matrices.values()))

        # grabbing the number of classes and query examples for easier use later in the function
        number_of_classes = class_means.size(0)
        number_of_targets = target_features.size(0)

        """
        Calculating the Mahalanobis distance between query examples and the class means
        including the class precision estimates in the calculations, reshaping the distances
        and multiplying by -1 to produce the sample logits
        """
        repeated_target = target_features.repeat(1, number_of_classes).view(-1, class_means.size(1))
        repeated_class_means = class_means.repeat(number_of_targets, 1)
        repeated_difference = (repeated_class_means - repeated_target)
        repeated_difference = repeated_difference.view(number_of_targets, number_of_classes,
                                                       repeated_difference.size(1)).permute(1, 0, 2)
        first_half = torch.matmul(repeated_difference, class_precision_matrices)
        sample_logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1, 0) * -1

        return sample_logits, class_means

    def _build_class_reps_and_covariance_estimates(self, context_features, context_labels, tied_covariance):
        """
        Construct and return class level representations and class covariance estimattes for each class in task.
        :param context_features: (torch.tensor) Adapted feature representation for each image in the context set.
        :param context_labels: (torch.tensor) Label for each image in the context set.
        :return: (void) Updates the internal class representation and class covariance estimates dictionary.
        """

        """
        Calculating a task level covariance estimate using the provided function.
        """
        for c in torch.unique(context_labels):
            # filter out feature vectors which have class c
            class_features = torch.index_select(context_features, 0, self._extract_class_indices(context_labels, c))
            # mean pooling examples to form class means
            class_rep = mean_pooling(class_features)
            # updating the class representations dictionary with the mean pooled representation
            self.class_representations[c.item()] = class_rep
            """
            Calculating the mixing ratio lambda_k_tau for regularizing the class level estimate with the task level estimate."
            Then using this ratio, to mix the two estimate; further regularizing with the identity matrix to assure invertability, and then
            inverting the resulting matrix, to obtain the regularized precision matrix. This tensor is then saved in the corresponding
            dictionary for use later in infering of the query data points.
            """
            self.class_precision_matrices[c.item()] = self.estimate_unsacled_cov(class_features)
        if tied_covariance:
            shared_cov = torch.stack(list(self.class_precision_matrices.values())).sum(0)/len(context_labels)
        for k,v in self.class_precision_matrices.items():
            self.class_precision_matrices[k] = torch.inverse(shared_cov).to(self.device) if tied_covariance else torch.inverse(v).to(self.device)

    def estimate_unsacled_cov(self, examples, rowvar=False, inplace=False):
        """
        Function based on the suggested implementation of Modar Tensai
        and his answer as noted in: https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/5

        Estimate a covariance matrix given data.

        Covariance indicates the level to which two variables vary together.
        If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
        then the covariance matrix element `C_{ij}` is the covariance of
        `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

        Args:
            examples: A 1-D or 2-D array containing multiple variables and observations.
                Each row of `m` represents a variable, and each column a single
                observation of all those variables.
            rowvar: If `rowvar` is True, then each row represents a
                variable, with observations in the columns. Otherwise, the
                relationship is transposed: each column represents a variable,
                while the rows contain observations.

        Returns:
            The covariance matrix of the variables.
        """
        if examples.dim() > 2:
            raise ValueError('m has more than 2 dimensions')
        if examples.dim() < 2:
            examples = examples.view(1, -1)
        if not rowvar and examples.size(0) != 1:
            examples = examples.t()
        if inplace:
            examples -= torch.mean(examples, dim=1, keepdim=True)
        else:
            examples = examples - torch.mean(examples, dim=1, keepdim=True)
        examples_t = examples.t()
        return  examples.matmul(examples_t).squeeze()




