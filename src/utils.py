import math
import os
import pprint
import shutil
import sys
import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import yaml
from attrdict import AttrDict
import numpy as np
import re
from torchvision import transforms
import glob


def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_path(path):
    if os.path.exists(path):
        if input('directory: {} exists, overwrite? ([y]/n)'.format(path)) == 'y':
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            sys.exit('Rename run-name and initiate new run')
    else:
        os.makedirs(path)
    return

def load_config_from_yaml(config_path)->AttrDict:
  print('Loaded configuration from file : {}'.format(config_path))
  # with open(config_path) as f:
  #     cfg = yaml.safe_load(f)  # config is dict
  cfg = AttrDict(parse_config(path=config_path))
  print(yaml.dump(dict(cfg), default_flow_style=False))
  return cfg

def config_to_printable(configs):
    return yaml.dump(dict(configs), default_flow_style=False)

def parse_config(path=None, data=None, tag='!ENV'):
    """
    Load a yaml configuration file and resolve any environment variables
    The environment variables must have !ENV before them and be in this format
    to be parsed: ${VAR_NAME}.
    E.g.:
    database:
        host: !ENV ${HOST}
        port: !ENV ${PORT}
    app:
        log_path: !ENV '/var/${LOG_PATH}'
        something_else: !ENV '${AWESOME_ENV_VAR}/var/${A_SECOND_AWESOME_VAR}'
    :param str path: the path to the yaml file
    :param str data: the yaml data itself as a stream
    :param str tag: the tag to look for
    :return: the dict configuration
    :rtype: dict[str, T]
    """
    # pattern for global vars: look for ${word}
    pattern = re.compile('.*?\${(\w+)}.*?')
    loader = yaml.SafeLoader

    # the tag will be used to mark where to start searching for the pattern
    # e.g. somekey: !ENV somestring${MYENVVAR}blah blah blah
    loader.add_implicit_resolver(tag, pattern, None)
    loader.add_implicit_resolver("!ENV_INT", pattern, None)
    loader.add_implicit_resolver("!ENV_FLOAT", pattern, None)

    def constructor_env_variables(loader, node):
        """
        Extracts the environment variable from the node's value
        :param yaml.Loader loader: the yaml loader
        :param node: the current node in the yaml
        :return: the parsed string that contains the value of the environment
        variable
        """
        value = loader.construct_scalar(node)
        match = pattern.findall(value)  # to find all env variables in line
        if match:
            full_value = value
            for g in match:
                full_value = full_value.replace(
                    f'${{{g}}}', os.environ.get(g, g)
                )
            return full_value
        return value

    def constructor_env_variables_int(loader, node):
        """
        Extracts the environment variable from the node's value
        :param yaml.Loader loader: the yaml loader
        :param node: the current node in the yaml
        :return: the parsed string that contains the value of the environment
        variable
        """
        value = loader.construct_scalar(node)
        match = pattern.findall(value)  # to find all env variables in line
        if match:
            full_value = value
            for g in match:
                full_value = full_value.replace(
                    f'${{{g}}}', os.environ.get(g, g)
                )
            return int(full_value)
        return value

    def constructor_env_variables_float(loader, node):
        """
        Extracts the environment variable from the node's value
        :param yaml.Loader loader: the yaml loader
        :param node: the current node in the yaml
        :return: the parsed string that contains the value of the environment
        variable
        """
        value = loader.construct_scalar(node)
        match = pattern.findall(value)  # to find all env variables in line
        if match:
            full_value = value
            for g in match:
                full_value = full_value.replace(
                    f'${{{g}}}', os.environ.get(g, g)
                )
            return float(full_value)
        return value

    loader.add_constructor(tag, constructor_env_variables)
    loader.add_constructor("!ENV_INT", constructor_env_variables_int)
    loader.add_constructor("!ENV_FLOAT", constructor_env_variables_float)

    if path:
        with open(path) as conf_data:
            return yaml.load(conf_data, Loader=loader)
    elif data:
        return yaml.load(data, Loader=loader)
    else:
        raise ValueError('Either a path or data should be defined as input')

def save_config_to_yaml(cfg,dir_path):
  with open(dir_path + '/exp_config.yaml', 'w') as outfile:
       yaml.dump(dict(cfg), outfile, default_flow_style=False)
  print('Saved configuration to directory : {}'.format(dir_path))

def ppconfigs(x, path):
    with open(path + '/run_configuration.txt', 'w') as f:
        f.write(pprint.pformat(vars(x)))
    pprint.pformat(vars(x))

def read_csv(path, names):
    return pd.read_csv(path, sep=',', names=names)

def save_all_py_files_in_src(source_path, destination_path, override=True):
    """
    Recursive copies files from source  to destination directory.
    :param source_path: source directory
    :param destination_path: destination directory
    :param override if True all files will be overridden otherwise skip if file exist
    :return: count of copied files
    """
    files_count = 0
    if not os.path.exists(destination_path):
        os.mkdir(destination_path)
    items = glob.glob(source_path+'/*')
    for item in items:
        if os.path.isdir(item):
            path = os.path.join(destination_path, item.split('/')[-1])
            files_count += save_all_py_files_in_src(source_path=item,destination_path=path, override=override)
        else:
            if item.endswith('.py'):
                file = os.path.join(destination_path, item.split('/')[-1])
                if not os.path.exists(file) or override:
                    shutil.copyfile(item, file)
                    files_count += 1
    return files_count


class ValidationAccuracies:
    """
    Determines if an evaluation on the validation set is better than the best so far.
    In particular, this handles the case for meta-dataset where we validate on multiple datasets and we deem
    the evaluation to be better if more than half of the validation accuracies on the individual validation datsets
    are better than the previous best.
    """

    def __init__(self, validation_datasets):
        self.datasets = validation_datasets
        self.dataset_count = len(self.datasets)
        self.current_best_accuracy_dict = {}
        for dataset in self.datasets:
            self.current_best_accuracy_dict[dataset] = {"accuracy": 0.0, "confidence": 0.0}

    def is_better(self, accuracies_dict):
        is_better = False
        is_better_count = 0
        for i, dataset in enumerate(self.datasets):
            if accuracies_dict[dataset]["accuracy"] > self.current_best_accuracy_dict[dataset]["accuracy"]:
                is_better_count += 1
        if self.dataset_count == is_better_count:
            is_better = True
        elif is_better_count >= int(math.ceil(self.dataset_count / 2.0)):
            is_better = True

        return is_better

    def replace(self, accuracies_dict):
        self.current_best_accuracy_dict = accuracies_dict

    def print(self, logfile, accuracy_dict):
        print_and_log(logfile, "")  # add a blank line
        print_and_log(logfile, "Validation Accuracies:")
        for dataset in self.datasets:
            print_and_log(logfile, "{0:}: {1:.1f}+/-{2:.1f}".format(dataset, accuracy_dict[dataset]["accuracy"],
                                                                    accuracy_dict[dataset]["confidence"]))
        print_and_log(logfile, "")  # add a blank line

    def get_current_best_accuracy_dict(self):
        return self.current_best_accuracy_dict


def verify_checkpoint_dir(checkpoint_dir, resume, test_mode, resume_epoch):
    if resume:  # verify that the checkpoint directory and file exists
        if not os.path.exists(checkpoint_dir):
            print("Can't resume for checkpoint. Checkpoint directory ({}) does not exist.".format(checkpoint_dir),
                  flush=True)
            sys.exit()

        checkpoint_file = os.path.join(checkpoint_dir, 'latest_checkpoint{}.pt'.format(resume_epoch))
        if not os.path.isfile(checkpoint_file):
            print("Can't resume for checkpoint. Checkpoint file ({}) does not exist.".format(checkpoint_file),
                  flush=True)
            sys.exit()
    elif test_mode:
        if not os.path.exists(checkpoint_dir):
            print("Can't test. Checkpoint directory ({}) does not exist.".format(checkpoint_dir), flush=True)
            sys.exit()
    else:
        # if train
        # exp_dir should already be created
        if not os.path.exists(checkpoint_dir):
            print("Checkpoint directory ({}) does not exits.".format(checkpoint_dir), flush=True)
            sys.exit()


def print_and_log(log_file, message):
    """
    Helper function to print to the screen and the cnaps_layer_log.txt file.
    """
    print(message, flush=True)
    log_file.write(message + '\n')


def get_log_files(checkpoint_dir):
    """
    Function that takes a path to a checkpoint directory and returns a reference to a logfile and paths to the
    fully trained model and the model with the best validation score.
    """
    # verify_checkpoint_dir(checkpoint_dir, resume, test_mode, resume_epoch)
    # if not test_mode and not resume:
    #     os.makedirs(checkpoint_dir)
    checkpoint_path_validation = os.path.join(checkpoint_dir, 'best_validation.pt')
    checkpoint_path_final = os.path.join(checkpoint_dir, 'fully_trained.pt')
    logfile_path = os.path.join(checkpoint_dir, 'log.txt')
    if os.path.isfile(logfile_path):
        logfile = open(logfile_path, "a", buffering=1)
    else:
        logfile = open(logfile_path, "w", buffering=1)

    return logfile, checkpoint_path_validation, checkpoint_path_final


def stack_first_dim(x):
    """
    Method to combine the first two dimension of an array
    """
    x_shape = x.size()
    new_shape = [x_shape[0] * x_shape[1]]
    if len(x_shape) > 2:
        new_shape += x_shape[2:]
    return x.view(new_shape)


def split_first_dim_linear(x, first_two_dims):
    """
    Undo the stacking operation
    """
    x_shape = x.size()
    new_shape = first_two_dims
    if len(x_shape) > 1:
        new_shape += [x_shape[-1]]
    return x.view(new_shape)


def sample_normal(mean, var, num_samples):
    """
    Generate samples from a reparameterized normal distribution
    :param mean: tensor - mean parameter of the distribution
    :param var: tensor - variance of the distribution
    :param num_samples: np scalar - number of samples to generate
    :return: tensor - samples from distribution of size numSamples x dim(mean)
    """
    sample_shape = [num_samples] + len(mean.size()) * [1]
    normal_distribution = torch.distributions.Normal(mean.repeat(sample_shape), var.repeat(sample_shape))
    return normal_distribution.rsample()


def loss(test_logits_sample, test_labels, device):
    """
    Compute the classification loss.
    """
    size = test_logits_sample.size()
    sample_count = size[0]  # scalar for the loop counter
    num_samples = torch.tensor([sample_count], dtype=torch.float, device=device, requires_grad=False)

    log_py = torch.empty(size=(size[0], size[1]), dtype=torch.float, device=device)
    for sample in range(sample_count):
        log_py[sample] = -F.cross_entropy(test_logits_sample[sample], test_labels, reduction='none')
    score = torch.logsumexp(log_py, dim=0) - torch.log(num_samples)
    return -torch.sum(score, dim=0)

def loss2(test_logits_sample, test_labels, device):
    """
    standard cross-entropy loss
    """
    return F.cross_entropy(test_logits_sample.squeeze(0), test_labels, reduction='mean')

def loss3(test_logits_sample, test_labels, device):
    """
    CD Margin loss for classification
    :param test_logits_sample:
    :param test_labels:
    :param device:
    :return:
    """
    pass


def aggregate_accuracy(test_logits_sample, test_labels):
    """
    Compute classification accuracy.
    """
    averaged_predictions = torch.logsumexp(test_logits_sample, dim=0)
    return torch.mean(torch.eq(test_labels, torch.argmax(averaged_predictions, dim=-1)).float())


def linear_classifier(x, param_dict):
    """
    Classifier.
    """
    return F.linear(x, param_dict['weight_mean'], param_dict['bias_mean'])


def setup_logging(name, output_dir, console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger(name)
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


### KL divergence ###
def gaussian_kl(q_mean, q_var, p_mean, p_var):
    '''
    compute KL(q||p) between two diagonal gaussian distributions
    assume tensor inputs
    All input have shape: (num_tasks, context_dim)
    '''
    # p_var = p_var.flatten()
    # q_var = q_var.flatten()
    # p_mean = p_mean.flatten()
    # q_mean = q_mean.flatten()
    return 0.5 * torch.sum(torch.log(p_var) - torch.log(q_var) - 1 + (p_mean - q_mean) ** 2 / p_var + q_var / p_var,
                           dim=-1)

def contrastive_loss(prior_latent, posterior_latent):
    '''
    :param latent_samples: shape (batch_size, latent_dim)
    :return:
    '''
    task_dist = torch.cdist(prior_latent.unsqueeze(0), posterior_latent.unsqueeze(0)).squeeze(0)
    return - (torch.log(torch.softmax(-task_dist, dim=1).diag()).mean() +
              torch.log(torch.softmax(-task_dist, dim=0).diag()).mean())


def gaussain_entrpy(q_var_batch):
    '''
    :param q_var_batch: shape (batch_size, latent_dim)
    :return:
    '''
    D = q_var_batch.shape[-1]
    return 0.5 * q_var_batch.log().sum(-1) + 0.5 * D * (1 + (2 * torch.tensor(torch.pi)).log())


##############################
#  KL coefficient scheduler  #
##############################
class KLSheduler():
    def __init__(self, mode):
        self.full_cycle = 2000
        self._mode = mode
        if self._mode == 'linear':
            self._beta = torch.tensor(0.0, device='cuda')
        else:
            self._beta = torch.tensor(1.0, device='cuda')

    def step(self):
        if self._mode == 'linear' and self._beta < 1.0:
            self._beta += 1 / self.full_cycle

    @property
    def beta(self):
        return self._beta


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



class toZeroOne:
    """Rotate by one of the given angles."""
    def __call__(self, x):
        return (x+1.0)/2.0

class toMinsOneOne:
    """Rotate by one of the given angles."""
    def __call__(self, x):
        return x*2.0-1.0

class rotate90:
    def __call__(self,x):
        angle = random.choice([90, 180, 270, 0])
        return transforms.functional.rotate(x, angle)



ood_transformation = transforms.Compose(
    [toZeroOne(),
    transforms.ToPILImage(),
    rotate90(),
    transforms.RandomChoice([transforms.ColorJitter(brightness=.5, hue=.3),
                              transforms.RandomPosterize(bits=1, p=1.0),
                              transforms.RandomPerspective(distortion_scale=0.5, p=1.0), ]),
    transforms.ToTensor(),
    toMinsOneOne()]
)






