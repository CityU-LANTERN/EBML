from src.ebml_models.modules import Conv4,DecoderBase, tsa_resnet18, MultiHeadAttentionLayer
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import os
from src.utils import loss,loss2,setup_logging,ood_transformation

FIXED_TESTSET_ROOT =os.environ['FIXED_TESTSET_ROOT']

def cosine_classifier(query_features, prototypes):
    query_features = query_features.unsqueeze(1)
    prototypes = prototypes
    return nn.functional.cosine_similarity(query_features, prototypes, dim=-1, eps=1e-30) * 10.0

def random_label_shuffle(labels, shuffle_ratio, shuffle_prob):
    if np.random.rand() < shuffle_prob:
        n_to_shuffle = np.floor(len(labels)*shuffle_ratio).astype(int)
        shuffled_index = np.random.permutation(n_to_shuffle)
        labels[:n_to_shuffle] = labels[shuffled_index]
    return labels

def random_input_augmentation(inputs, aug_ratio,aug_prob):
    if np.random.rand() < aug_prob:
        inputs = torch.stack([ood_transformation(x).cuda() for x in inputs])
    return inputs


class PriorEBM(nn.Module):
    def __init__(self, configs):
        super(PriorEBM, self).__init__()
        assert configs.model.mha.mode in ['basic','basic+rff']
        self.mha_mode = configs.model.mha.mode
        self.feat_dim = 800 if configs.model.feature_extractor=='conv4' else 512
        sn = nn.utils.spectral_norm if configs.model.prior_sn else lambda x: x
        self.energy_net = MultiHeadAttentionLayer(d_model=configs.model.mha.feature_dim,
                                                  num_heads=configs.model.mha.num_heads,
                                                  sn=sn)
        self.layernorm1 = nn.LayerNorm([self.feat_dim], elementwise_affine=False)
        self.energy_net2 = nn.Sequential(sn(nn.Linear(512, 256)),
                                        nn.SiLU(),
                                         sn(nn.Linear(256, 1)))
        if configs.model.mha.mode =='basic+rff':
            self.rff = nn.Sequential(sn(nn.Linear(512,512)),
                                     nn.SiLU())
            self.layernorm2 = nn.LayerNorm([self.feat_dim], elementwise_affine=False)
        if not configs.training:
            self.eval()

    def forward(self, task_latent):
        # x = self.energy_net(task_latent)
        x = self.layernorm1(self.energy_net(task_latent, task_latent, task_latent) + task_latent)
        if self.mha_mode =='basic+rff':
            x = self.layernorm2(self.rff(x) + x)
        x = x.mean(-2)
        x = self.energy_net2(x)
        return x


class GenerativeClassifier(DecoderBase):
    def __init__(self, configs):
        super(GenerativeClassifier, self).__init__(configs)
        self.base_energy_f = None
        self.base_energy_weight = configs.model.base_energy_weight
        self.sn = configs.model.decoder_sn
        feat_dim = 512
        sn = nn.utils.spectral_norm if self.sn else lambda x: x
        self.e_net = nn.Sequential(
            sn(nn.Linear(feat_dim + feat_dim , 512)), nn.SiLU(),
            sn(nn.Linear(512, 512)), nn.SiLU(),
            sn(nn.Linear(512, 512)), nn.SiLU(),
            sn(nn.Linear(512, 1))
        )

    def forward(self, features, latent):
        assert latent.shape == features.shape, \
                f'Evaluating E at every pair of xy require matching shape of features and latent. ' \
                f'But received features : {features.shape} and latent : {latent.shape}'

        e_xy_base = self.e_net(torch.cat([features, latent],dim=-1)).squeeze(-1)
        return e_xy_base


class EBMLMetaClassifier(nn.Module):
    def __init__(self, configs):
        super(EBMLMetaClassifier, self).__init__()
        self.configs = configs
        # config modules
        self.device = configs.device
        self.task_ebm = GenerativeClassifier(configs)
        self.prior_ebm = PriorEBM(configs)

        if hasattr(configs.model,'feature_extractor') and configs.model.feature_extractor =='resnet':
            tsa_wrapper = self.configs.tta.num_steps > 0 and not configs.training
            self.feature_extractor = tsa_resnet18(pretrained=configs.model.use_pretrained_backbone,
                                                      pretrained_model_path=configs.model.pretrained_resnet_path,
                                                      training=False, #freeze the pre-trained backbone as in TSA
                                                      tsa_wrapper=tsa_wrapper,
                                                      configs=configs)
            if not self.configs.training:
                # used to produce baseline reuslts only
                self.feature_extractor_url_pretrained = tsa_resnet18(pretrained=configs.model.use_pretrained_backbone,
                                                      pretrained_model_path=configs.model.pretrained_resnet_path,
                                                      training=False,
                                                      tsa_wrapper=False,
                                                      configs=configs)
        else:
            self.feature_extractor = Conv4(configs)

        if configs.loss.use_ce:
            self.loss_func = loss2
        else:
            self.loss_func = loss
        # Buffers for EBM
        self.latent_buffer_size = int(5000)
        self.task_buffer_size = int(10000 / configs.model.num_phi_samples)
        if configs.sgld.decoder.init == 'buffer':
            self.task_buffer = torch.rand(size=(configs.model.num_phi_samples, self.task_buffer_size, 1024), dtype=torch.float32)
        if configs.sgld.prior.init == 'buffer':
            self.latent_buffer = torch.randn(size=(self.latent_buffer_size, configs.model.latent_dim), dtype=torch.float32)
        # debug options
        self.iteration = 0
        self.verbose = configs.train.debug
        if self.verbose:
            self.debug_logger = setup_logging('debug', configs.exp_dir)
        # task loss dict
        self.loss_dict = {}
        self.phi_samples = []

        if configs.training:
            param_sets = [{'params': self.task_ebm.parameters(), 'lr': configs.optimizer.lr_ebm},
                          {'params': self.prior_ebm.parameters(), 'lr': configs.optimizer.lr_ebm}]
            self.feature_extractor.requires_grad_(False)
            self.optimizer = torch.optim.Adam(param_sets, lr=configs.optimizer.lr_vae,
                                              betas=(configs.optimizer.beta1, configs.optimizer.beta2),
                                              weight_decay=configs.optimizer.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                             gamma=configs.scheduler.gamma,
                                                             step_size=configs.scheduler.decay_every_step)
        self.beta_distribution = torch.distributions.beta.Beta(torch.tensor([2.0]), torch.tensor([5.0]))

    def validate_loss(self):
        task_loss = torch.stack(list(self.loss_dict.values()))
        values = task_loss.detach().tolist()
        keys = list(self.loss_dict.keys())
        self.loss_dict.clear()
        return task_loss.sum(), (keys, values)

    def shuffle(self, images, labels):
        """
        Return shuffled data.
        """
        permutation = np.random.permutation(images.shape[0])
        return images[permutation], labels[permutation]

    def prepare_task(self, task_dict, shuffle=True):
        '''
        :param task_dict: dict containing support and query images and labels
        :return: support and query splits for an input task,  in 4D tensor of shape (num_supp/num_query, C,H,W)
        '''
        context_images_np, context_labels_np = task_dict['context_images'], task_dict['context_labels']
        context_images_np = context_images_np.transpose([0, 3, 1, 2])
        if shuffle :
            context_images_np, context_labels_np = self.shuffle(context_images_np, context_labels_np)
        context_images = torch.from_numpy(context_images_np).to(self.device).float()
        context_labels = torch.from_numpy(context_labels_np).to(self.device).long()
        target_images_np, target_labels_np = task_dict['target_images'], task_dict['target_labels']
        target_images_np = target_images_np.transpose([0, 3, 1, 2])
        if shuffle:
            target_images_np, target_labels_np = self.shuffle(target_images_np, target_labels_np)
        target_images = torch.from_numpy(target_images_np).to(self.device).float()
        target_labels = torch.from_numpy(target_labels_np).to(self.device).long()
        return (context_images, context_labels), (target_images, target_labels)

    def meta_update(self, task_dict):
        '''
        Run one learning iteration and return meta loss for a task, assuming batch_size is always 1
        '''
        (x_s, y_s), (x_q, y_q) = self.prepare_task(task_dict)
        acc = 0.0
        ####################
        '''classification'''
        ####################
        id_s_features = self.feature_extractor(x_s).unsqueeze(0)
        _, phi_samples_id = self.task_ebm.adapt_features_prototypes(id_s_features, y_s, phi=None)
        id_q_features = self.feature_extractor(x_q)
        mean_logits = cosine_classifier(id_q_features,phi_samples_id)
        with torch.no_grad():
            y_pred = torch.argmax(mean_logits, -1)
            acc = (y_q == y_pred).sum().item() / y_q.numel()
        #################
        '''ebms learning '''
        #################
        # sample task representation
        x_s_neg = random_input_augmentation(x_s,aug_ratio = 0.6, aug_prob = 0.9)
        neg_features = self.feature_extractor(x_s_neg).unsqueeze(0)
        _, phi_samples_neg = self.task_ebm.adapt_features_prototypes(neg_features, y_s, phi=None)
        phi_samples_neg, accumualted_grad_norm = self.prior_sgld(phi_samples_neg)
        epos_prior = self.prior_ebm(phi_samples_id+torch.randn_like(phi_samples_id)*1e-3)
        eneg_prior = self.prior_ebm(phi_samples_neg.detach())
        # prior cd loss
        prior_cd_loss = epos_prior.mean() - eneg_prior.mean()
        assert prior_cd_loss.isfinite()
        # data cd loss
        phi_samples_id_yq = torch.index_select(phi_samples_id,index=y_q.flatten(),dim=1)
        id_q_features = id_q_features.unsqueeze(0)
        assert phi_samples_id_yq.shape==id_q_features.shape, (phi_samples_id_yq.shape,id_q_features.shape)
        epos_task = self.task_ebm(id_q_features,phi_samples_id_yq)
        neg_q_features, phi_samples_neg_yq = self.task_sgld(id_q_features,phi_samples_id)
        eneg_task = self.task_ebm(neg_q_features,phi_samples_neg_yq)
        task_cd_loss = epos_task.mean() - eneg_task.mean()
        assert task_cd_loss.isfinite()
        #######################
        '''energy l2 penalty'''
        #######################
        if self.configs.loss.l2_weight > 0:
            l2 = epos_prior.square().mean() + eneg_prior.square().mean()
            l2 += epos_task.square().mean() + eneg_task.square().mean()
        else:
            l2 = torch.tensor(0, device=self.device)
        ################
        '''meta loss'''
        ################
        # self.loss_dict['CE_ood_loss'] = xentropy_ood * 0.5
        self.loss_dict['prior_cd_loss'] = prior_cd_loss
        self.loss_dict['task_cd_loss'] = task_cd_loss
        self.loss_dict['ebm_l2'] = l2 * self.configs.loss.l2_weight
        #for recording only, no loss gradients
        self.loss_dict['epos_prior'] = epos_prior.detach().mean()
        self.loss_dict['eneg_prior'] = eneg_prior.detach().mean()
        self.loss_dict['epos_task'] = epos_task.detach().mean()
        self.loss_dict['eneg_task'] = eneg_task.detach().mean()
        self.loss_dict['accumualted_grad_norm'] = accumualted_grad_norm.detach()
        meta_loss, loss_discriptor = self.validate_loss()
        self.iteration += 1

        return meta_loss, \
               loss_discriptor, \
               acc

    def task_sgld(self, features, latents):
        features = features.detach()[:,:,None,:].repeat(1,1,latents.size(1),1)
        features += torch.randn_like(features) * 1e-2
        latents = latents.detach()[:,None,:,:].repeat(1,features.size(1),1,1)
        latents += torch.randn_like(latents) * 1e-2
        features.requires_grad = True
        latents.requires_grad = True
        features.data = torch.clamp(features, min=0.0)
        latents.data = torch.clamp(latents, min=0.0)
        for i in range(self.configs.sgld.decoder.niter):
            e_xyz = self.task_ebm(features,latents).sum()
            grad_x, grad_yz = torch.autograd.grad(e_xyz, [features, latents], retain_graph=True)
            features.data -= self.configs.sgld.decoder.step * grad_x + self.configs.sgld.decoder.eta * torch.randn_like(features)
            latents.data -= self.configs.sgld.decoder.step * grad_yz + self.configs.sgld.decoder.eta * torch.randn_like(latents)
            features.data = torch.clamp(features, min=0.0)  # since we used relu in resnet-18
            latents.data = torch.clamp(latents, min=0.0) # since we used relu in resnet-18
        return features.detach(), latents.detach()
            
    def prior_sgld(self, features):
        accumulated_gradx = torch.tensor(0.0, device=self.device)
        beta = self.beta_distribution.sample(sample_shape=(1, features.shape[1])).to(self.device)
        features = torch.randn_like(features) * beta + (1 - beta) * features.detach()
        self.prior_ebm.eval()
        features.requires_grad = True

        for i in range(self.configs.sgld.prior.niter):
            e_x = self.prior_ebm(features).mean()
            grad = torch.autograd.grad(e_x, features, retain_graph=True)[0]
            features.data -= self.configs.sgld.prior.step * grad + self.configs.sgld.prior.eta * torch.randn_like(
                features)
            features.data = torch.clamp(features, min=0.0)
            accumulated_gradx += torch.norm(grad, dim=-1).mean()
        self.prior_ebm.train()
        return features.detach(), accumulated_gradx

    def gradient_surgery(self,loss_list,param_list):
        grad_list = [torch.cat([grad.flatten() for grad in torch.autograd.grad(l, param_list, retain_graph=True)]) for l in loss_list]
        proj_grad_list = deepcopy(grad_list)
        # zero grad
        for p in param_list:
            p.grad = torch.zeros_like(p)
        # iterate through all gradients
        for proj_grad in proj_grad_list:
            # projection
            # random.shuffle(grad_list)
            # for grad_j in grad_list:
            #     dot_prod = torch.dot(proj_grad,grad_j)
            #     if dot_prod <0 :
            #         proj_grad -= dot_prod * grad_j / (grad_j.norm().square())
            # accumulate gradients
            idx = 0
            for p in param_list:
                length = torch.numel(p)
                p.grad += proj_grad[idx:idx + length].view(p.shape).clone()
                idx += length
            assert len(proj_grad) == idx

    def test_time_adaptation(self, x_s, y_s, x_q, y_q,num_steps,lr_alpha,lr_beta,energy_weight,ce_weight,entropy_weight) -> dict:
        tta_records = {'energy_trajectory': [],
                       'ce_loss_trajectory': [],
                       'entropy_q_trajectory':[],
                       'q_accuracy_trajectory':[],
                       'q_ce_loss_trajectory':[],
                       'q_energy_trajectory':[],
                       'sq_pseudo_ce_loss_trajectory':[],
                       's_prototypes':[],
                       'sq_pseudo_prototypes':[],
                       'grad_cos_similarity':[],
                       'grad_mag_similarity': []}
        alpha_params = {k:v for k, v in self.feature_extractor.named_parameters() if 'alpha' in k}
        beta_params = {k:v for k, v in self.feature_extractor.named_parameters() if 'beta' in k}
        tta_param_list = []
        if 'alpha' in self.configs.tta.tsa.opt:
            tta_param_list.append({'params': alpha_params.values(), 'lr': lr_alpha})
        if 'beta' in self.configs.tta.tsa.opt:
            tta_param_list.append({'params': beta_params.values(), 'lr': lr_beta})
        if energy_weight:
            opt = torch.optim.Adam(tta_param_list, lr=self.configs.tta.lr)
        else:
            opt = torch.optim.Adadelta(tta_param_list, lr=self.configs.tta.lr)
        
        for i in torch.arange(num_steps, device=self.device):
            if 'alpha' in self.configs.tta.tsa.opt:
                s_features = self.feature_extractor(x_s)
                q_features = self.feature_extractor(x_q)
            if 'beta' in self.configs.tta.tsa.opt:
                s_features = self.feature_extractor.beta(s_features)
                q_features = self.feature_extractor.beta(q_features)
            _,phi_samples = self.task_ebm.adapt_features_prototypes(s_features.unsqueeze(0),y_s,phi=None)

            s_logits = cosine_classifier(s_features,phi_samples)
            ce_loss = self.loss_func(test_logits_sample=s_logits.unsqueeze(0), test_labels=y_s.view(-1),
                                      device=self.device)
            tta_records['ce_loss_trajectory'].append(ce_loss.item())
            with torch.no_grad():
                _, phi_samples_q = self.task_ebm.adapt_features_prototypes(q_features.unsqueeze(0), y_q, phi=None)
                tta_records['q_energy_trajectory'].append(self.prior_ebm(phi_samples_q).mean().item())

                q_logits = cosine_classifier(q_features, phi_samples)
                y_q_pred = q_logits.argmax(-1)
                tta_records['q_accuracy_trajectory'].append((y_q == y_q_pred).float().mean().item())
                xentropy = self.loss_func(test_logits_sample=q_logits.unsqueeze(0), test_labels=y_q.view(-1),
                                          device=self.device)
                tta_records['q_ce_loss_trajectory'].append(xentropy.item())

            sq_features = torch.cat([s_features,q_features],dim=0)
            sq_y = torch.cat([y_s,y_q_pred])
            _, phi_samples_sq = self.task_ebm.adapt_features_prototypes(sq_features.unsqueeze(0), sq_y, phi=None)
            energy = self.prior_ebm(phi_samples_sq).mean() # todo phi_samples back to phi_samples_sq
            energy_loss = energy + self.configs.tta.energy_m
            tta_records['energy_trajectory'].append(energy.item())
            opt.zero_grad()
            if energy_loss<=0.0:
                energy_loss = energy_loss * torch.zeros_like(energy_loss)
            self.gradient_surgery(
                loss_list=[ce_weight*ce_loss,energy_weight*energy_loss],
                param_list=list(alpha_params.values())+list(beta_params.values()))
            opt.step()

        return tta_records

    def begin_tta(self):
        self.feature_extractor.reset()
        self.feature_extractor.eval()
        return None

    def reset_tta(self, original_state):
        pass

    def evaluate_batch(self, task_dict, num_phi_samples, return_ood_scores=False) -> dict:
        task_results = {'no_tta': {}}
        tta_settings =[None]
        if not self.configs.training:
            settings = [self.configs.tta.num_steps,
                                self.configs.tta.tsa.lr_alpha,
                                self.configs.tta.tsa.lr_beta,
                                self.configs.tta.energy_weight,
                                self.configs.tta.ce_weight,
                                self.configs.tta.entropy_weight]
            if settings[0]: 
                task_results.update({"S{0}_A{1}_B{2}_PE{3}_CE{4}_EQ{5}_M{6}".format(*settings,self.configs.tta.energy_m).replace('.','f'): {}})
                tta_settings.append(settings)

        (x_s, y_s), (x_q, y_q) = self.prepare_task(task_dict)
        # test_time adaptation
        for i, (tta_mode,settings) in enumerate(zip(task_results.keys(),tta_settings)):
            log_dict = task_results.get(tta_mode)

            if tta_mode != 'no_tta':
                feature_extractor = self.feature_extractor
                original_state = self.begin_tta()
                log_dict.update(self.test_time_adaptation(x_s, y_s, x_q,y_q, *settings))
            else:
                feature_extractor = self.feature_extractor_url_pretrained

            with torch.no_grad():
                s_features = feature_extractor(x_s)
                q_features = feature_extractor(x_q)
                if tta_mode != 'no_tta' and 'beta' in self.configs.tta.tsa.opt:
                    s_features = feature_extractor.beta(s_features)
                    q_features = feature_extractor.beta(q_features)

                _, phi_samples = self.task_ebm.adapt_features_prototypes(s_features.unsqueeze(0), y_s, phi=None)
                log_dict['phi'] = phi_samples.squeeze(0).detach().cpu()
                # log p(y|x)
                mean_logits = cosine_classifier(q_features,phi_samples)
                y_q_pred = mean_logits.argmax(-1)
                task_acc = (y_q == y_q_pred).float().mean().item()
                log_dict['accuracy'] = task_acc
                # detect ood based on the support 
                if return_ood_scores:
                    supp_logits = cosine_classifier(s_features,phi_samples)
                    # '''-logits of classifier'''
                    log_dict['max logit'] = -supp_logits.max(dim=-1)[0].mean().item()
                    # '''-max softmax'''
                    supp_prob = nn.functional.softmax(supp_logits, dim=-1)
                    max_softmax, _ = supp_prob.max(dim=-1)
                    log_dict['max softmax'] = -max_softmax.mean().item()
                    # '''entropy'''
                    plogp = supp_prob * torch.log(supp_prob + 1e-8)
                    entropy = -plogp.sum(-1)
                    log_dict['entropy'] = entropy.mean().item()
                    # '''prior e'''
                    e_prior = self.prior_ebm(phi_samples)
                    log_dict['prior ebm'] = e_prior.mean().item()
                    e_task = self.task_ebm(s_features.unsqueeze(0),torch.index_select(phi_samples,index=y_s.flatten(),dim=1))
                    log_dict['task ebm'] = e_task.mean().item()
                    # ''' sum prior e and entropy '''
                    log_dict['sum energy entropy'] = e_prior.mean().item() + e_task.mean().item()
            if tta_mode != 'no_tta':
                self.reset_tta(original_state)
        return task_results

    def load_state_dict(self, state_dict:dict,strict: bool = True):
        print(f'{f"ignoring feature extrator (URL ResNet18)" :=^50}')
        self.prior_ebm.load_state_dict(state_dict['prior_ebm'],strict=True)
        if state_dict.get('task_ebm'):
            self.task_ebm.load_state_dict(state_dict['task_ebm'],strict=True)
        else:
            print(f'{f"state_dict does not have attribute : task_ebm " :=^50}')

    def state_dict(self):
        return {
            'feature_extractor_hyper_net' : self.feature_extractor.state_dict(),
            'prior_ebm' : self.prior_ebm.state_dict(),
            'task_ebm' : self.task_ebm.state_dict(),
        }

   