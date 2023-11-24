from src.utils import *
import torch
import torch.nn as nn
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
from copy import deepcopy
import scipy.stats
from collections import OrderedDict
from src.utils import set_random_seed
matplotlib.use("Agg")

#############################
#  Models for Regression   #
#############################
class Encoder(nn.Module):
    '''
    Compute task-specific context distribution, phi_i ~ N(encoder_mean(x,y),encoder_var(x,y))
    for each task in a input task batch of shape: (num_tasks,num_samples_per_task,sample_dim)
    '''

    def __init__(self, configs):
        super().__init__()
        self.hid_dim = configs.model.hid_dim
        self.latent_dim = configs.model.latent_dim
        # sn = nn.utils.parametrizations.spectral_norm
        sn = lambda x: x
        self.shared_encode = nn.Sequential(
            sn(nn.Linear(configs.dataset.dim_y + configs.dataset.dim_w, self.hid_dim)),
            nn.BatchNorm1d(self.hid_dim,track_running_stats=False),
            nn.ReLU(),
            sn(nn.Linear(self.hid_dim, self.hid_dim)),
            nn.BatchNorm1d(self.hid_dim, track_running_stats=False),
            nn.ReLU(),
            sn(nn.Linear(self.hid_dim, self.hid_dim)),
            nn.BatchNorm1d(self.hid_dim, track_running_stats=False),
            nn.ReLU(),
        )
        self.log_sigma = nn.Sequential(
            sn(nn.Linear(self.hid_dim, self.hid_dim)),
            nn.ReLU(),
            sn(nn.Linear(self.hid_dim, self.hid_dim)),
            nn.ReLU(),
            sn(nn.Linear(self.hid_dim, self.latent_dim)),
        )
        self.mu = nn.Sequential(
            sn(nn.Linear(self.hid_dim, self.hid_dim)),
            nn.ReLU(),
            sn(nn.Linear(self.hid_dim, self.hid_dim)),
            nn.ReLU(),
            sn(nn.Linear(self.hid_dim, self.latent_dim)),
        )

    def forward(self, x, y):
        '''
        :param x: shape (batch_size,task_dim,sample_dim)
        :param y: shape (batch_size,task_dim,sample_dim)
        :return: phi_mean: shape (batch_size,task_latent_dim)
                phi_std : shape (batch_size,task_latent_dim)
        '''
        # concat x and y along last dimension
        xy = torch.cat([x, y], dim=-1)  # shape: (batch_size, task_dim, 2)
        # forward pass and take average in each task
        latent = self.shared_encode(xy).mean(-2)
        # mu and sigma
        latent_mean = self.mu(latent)
        latent_std = self.log_sigma(latent).exp()
        return latent_mean, latent_std


class EBMDecoderPX(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.hid_dim = configs.model.hid_dim
        self.latent_dim = configs.model.latent_dim
        self.xy_input = configs.model.pxy_decoder
        sn = nn.utils.parametrizations.spectral_norm if configs.model.decoder_sn else lambda x: x
        act_fun = nn.SiLU
        input_dim =  1024 + configs.model.pxy_decoder
        self.fc_energy = nn.Sequential(
            sn(nn.Linear(self.latent_dim + input_dim, self.hid_dim)),
            act_fun(),
            sn(nn.Linear(self.hid_dim, self.hid_dim)),
            act_fun(),
            sn(nn.Linear(self.hid_dim, self.hid_dim)),
            act_fun(),
            sn(nn.Linear(self.hid_dim, 1))
        )

    def forward(self, x, y, phi):
        '''
        :param x: shape (batch_size,task_dim,sample_dim)
        :param y: shape (batch_size,task_dim,sample_dim)
        :param phi: shape (num_phi_samples,batch_size,task_latent_dim)
        :return: e: shape (num_phi_samples,batch_size,sample_dim, 1), energy
        '''
        num_phi_samples = phi.shape[0]
        x = x.unsqueeze(0).expand(num_phi_samples, -1, -1)
        z = phi.unsqueeze(-2).expand(-1, x.shape[-2], -1)
        if self.xy_input:
            y = y.unsqueeze(0).expand(num_phi_samples, -1, -1)
            e_xyz = self.fc_energy(torch.cat([x, y, z], dim=-1))  # shape ( batch_size, task_dim ,1)
        else:
            e_xyz = self.fc_energy(torch.cat([x, z], dim=-1))  # shape ( batch_size, task_dim ,1)
        return e_xyz


class Prior(nn.Module):
    def __init__(self, configs):
        super(Prior, self).__init__()
        sn = nn.utils.parametrizations.spectral_norm if configs.model.prior_sn else lambda x: x
        self.fc = nn.Sequential(
            nn.Linear(configs.model.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, phi):
        return self.fc(phi)


class GaussianDecoderPY(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.hid_dim = configs.model.hid_dim
        self.latent_dim = configs.model.latent_dim
        self.input_dim = 1
        self.mlp = nn.Sequential(
            nn.Linear(configs.dataset.dim_w + self.latent_dim, self.hid_dim),
            nn.LayerNorm([self.hid_dim]),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.LayerNorm([self.hid_dim]),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.LayerNorm([self.hid_dim]),
            nn.ReLU(),
            nn.Linear(self.hid_dim, 2)
        )

    def forward(self, x, y, phi):
        '''
        CNP classifier
        '''
        num_phi_samples = phi.shape[0]
        y = y.unsqueeze(0).expand(num_phi_samples, -1, -1)
        x = x.unsqueeze(0).expand(num_phi_samples, -1, -1)
        z = phi.unsqueeze(-2).expand(-1, x.shape[-2], -1)
        xz = torch.cat([x, z], dim=-1)
        y_pred = self.mlp(xz)
        # compute gaussian-like energy score @ y
        e_xy = 0.5 * ((y[:, :, 0:1] - y_pred[:, :, 0:1]) / y_pred[:, :, 1:2].exp()).square() + y_pred[:, :,1:2]
        return e_xy, y_pred


class EBMMetaRegressor(nn.Module):
    def __init__(self, configs):
        super(EBMMetaRegressor, self).__init__()
        self.configs = configs
        # self.configs.meta_batch_size=4
        self.encoder = Encoder(configs)
        self.decoder_py = GaussianDecoderPY(configs)
        self.ebmprior = Prior(configs)
        self.device = configs.device
        self.iteration = 0
        # projection head for latent variable
        self.projection_head = nn.Linear(configs.model.latent_dim, 32).cuda()
        self.con_loss = SupConLoss(temperature=0.5)
        # optimizers
        param_list =[{'params': self.encoder.parameters()},
                    {'params': self.decoder_py.parameters()},
                    {'params': self.projection_head.parameters()},
                    {'params': self.ebmprior.parameters()}]
        if configs.model.px_decoder or configs.model.pxy_decoder:
            self.decoder_px = EBMDecoderPX(configs)
            param_list.append({'params': self.decoder_px.parameters()})
        self.optimizer = torch.optim.Adam(param_list,
                                          lr=configs.optimizer.lr,
                                          betas=(configs.optimizer.beta1, configs.optimizer.beta2))
        if configs.scheduler.name:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=configs.scheduler.gamma,
                                                             step_size=configs.scheduler.decay_every_step)
        # sgld buffers
        self.prior_buffer = torch.randn(10000, self.configs.model.latent_dim, dtype=torch.float32).cuda()
        self.loss_dict = OrderedDict()

    def refine_phi(self,x_s,y_s,phi):
        # first order
        phi_clone = phi.clone().detach()
        phi_clone.requires_grad = True
        for i in range(5):
            epos, _ = self.decoder_py(x_s, y_s, phi_clone)
            vae_cd_loss = epos.mean()
            grad = torch.autograd.grad(vae_cd_loss,phi_clone)[0]
            phi_clone = phi_clone - 0.01*grad
        phi_clone = phi_clone - phi.detach() + phi # re-attach gradients to phi
        return phi_clone


    def meta_update(self, x_s,y_s,x_q,y_q):
        """
        compute training batch loss and update learner parameters
        """
        #####################
        '''ebm prior loss'''
        #####################
        # sample posterior phi from amortized encoder
        phi_s_batch, q_s_mean, q_s_std = self.sample_latent_from_q(x_s, y_s, self.configs.model.num_phi_samples)
        # phi_s_batch = self.refine_phi(x_s,y_s,phi_s_batch)
        # sample prior phi from ebm prior model
        if not self.configs.sgld.prior.sample_in_xy:
            ############### sample in phi space ###############
            phi_s_ebm, prior_sgld_summary = self.sample_latent_from_prior(phi_s_batch, return_summary=True)
        else:
            ############### sample in xy space ###############
            x_, y_, prior_sgld_summary = self.sample_xy_from_prior(x_s, y_s, return_summary=True)
            phi_s_ebm, _, _ = self.sample_latent_from_q(x_, y_, self.configs.model.num_phi_samples)
        # compute learning loss wrt ebm prior
        e_phi_pos = self.ebmprior(phi_s_batch)
        e_phi_neg = self.ebmprior(phi_s_ebm)  # shape : (num_samples,1)
        prior_cd_loss = e_phi_pos.mean() - e_phi_neg.mean()
        assert not torch.isnan(prior_cd_loss)
        self.loss_dict.update(prior_sgld_summary)
        ######################
        '''contrastive loss'''
        ######################
        if self.configs.loss.contrastive_loss:
            phi_q_batch, _, _ = self.sample_latent_from_q(x_q, y_q, self.configs.model.num_phi_samples)
            z_matrix = torch.concat([phi_s_batch, phi_q_batch], dim=0)  # 2*phi_samples, batch_size, latent_dim
            z_matrix = self.projection_head(z_matrix)
            z_matrix = F.normalize(z_matrix, dim=-1)
            self.loss_dict['contrasive loss'] = self.con_loss(torch.permute(z_matrix, [1, 0, 2]))
        ########################
        '''vae learning loss'''
        ########################
        if self.configs.model.deterministic_phi:
            # regularize the l2 norm of the latent variable
            self.loss_dict['l2 phi'] = torch.norm(phi_s_batch, dim=-1, p=2).mean() * self.configs.loss.kl_weight
        else:
            kl_loss = gaussian_kl(q_s_mean,
                                  q_s_std ** 2,
                                  torch.zeros_like(q_s_mean),
                                  torch.ones_like(q_s_std)).mean()
            self.loss_dict['kl loss'] = kl_loss * self.configs.loss.kl_weight
        # gaussian prediction p(y|x)
        pyx_nll_g, y_pred_g = self.decoder_py(x_q, y_q, phi_s_batch)
        pyx_nll_g = pyx_nll_g.mean()
        if self.configs.model.px_decoder or self.configs.model.pxy_decoder:
            # EBM p(x|phi)
            x_sq =torch.cat([x_q,x_s])
            y_sq = torch.cat([y_q,y_s])
            x_sq_neg,y_sq_neg,xy_sgld_summary= self.sample_xy_from_posterior(x_sq, y_sq, phi_s_batch,
                                                                             niter=self.configs.sgld.decoder.niter,
                                                                             return_summary=True,
                                                                             sample_x=True,
                                                                             sample_y=True)
            e_x_pos = self.decoder_px(x_sq, y_sq, phi_s_batch)
            e_x_neg = self.decoder_px(x_sq_neg, y_sq_neg, phi_s_batch)
            px_cd_loss = e_x_pos.mean() - e_x_neg.mean()
            self.loss_dict.update(xy_sgld_summary)
            assert not torch.isnan(px_cd_loss)
        query_mse_g = nn.functional.mse_loss(y_pred_g[:, :, 0:1].mean(0), y_q).detach()
        self.loss_dict['train_mse_g'] = query_mse_g
        # EBM p(y|x,phi)
        if self.configs.model.pxy_decoder and self.configs.model.refine_gaussian:
            x_q_repeat, y_pred_e, y_sgld_summary = self.sample_xy_from_posterior(x_q, y_pred_g[:, :, 0:1].mean(0), phi_s_batch,
                                                                        niter=self.configs.model.refine_gaussian,
                                                                        return_summary=True,
                                                                        sample_x=False,
                                                                        sample_y=True,
                                                                        py0_std=y_pred_g[:, :, 1:2].exp().mean(0))
            y_pred_e_mean = y_pred_e.reshape(self.configs.sgld.decoder.over_sample_ratio, len(y_q),1).mean(0)
            query_mse_e = nn.functional.mse_loss(y_pred_e_mean, y_q).detach()
            self.loss_dict['train_mse_refined'] = query_mse_e
            self.loss_dict.update(y_sgld_summary)
            e_y_pos = self.decoder_px(x_q,y_q,phi_s_batch)
            e_y_neg = self.decoder_px(x_q_repeat, y_pred_e, phi_s_batch)
            py_cd_loss = e_y_pos.mean() - e_y_neg.mean()
            assert not torch.isnan(py_cd_loss)

        ################
        '''l2 energy'''
        ################
        if self.configs.loss.l2_weight > 0:
            self.loss_dict['l2 energy'] = (e_phi_pos.square().mean() + e_phi_neg.square().mean()) \
                                          * self.configs.loss.l2_weight
            self.loss_dict['e prior pos'] = e_phi_pos.detach().mean()
            self.loss_dict['e prior neg'] = e_phi_neg.detach().mean()
            if self.configs.model.px_decoder or self.configs.model.pxy_decoder:
                self.loss_dict['l2 energy'] += (e_x_pos.square().mean() + e_x_neg.square().mean()) * self.configs.loss.l2_weight
                self.loss_dict['e decoder pos'] = e_x_pos.detach().mean()
                self.loss_dict['e decoder neg'] = e_x_neg.detach().mean()
                if self.configs.model.refine_gaussian:
                    self.loss_dict['l2 energy'] += (e_y_pos.square().mean() + e_y_neg.square().mean()) * self.configs.loss.l2_weight
                    self.loss_dict['e decoder y|x pos'] = e_y_pos.detach().mean()
                    self.loss_dict['e decoder y|x neg'] = e_y_neg.detach().mean()
        ################
        '''meta-loss'''
        ################
        if self.configs.model.px_decoder or self.configs.model.pxy_decoder:
            self.loss_dict['p(x|phi) cd loss'] = px_cd_loss * self.configs.loss.px_weight
            if self.configs.model.refine_gaussian:
                self.loss_dict['p(y|x,phi) cd loss'] = py_cd_loss
        self.loss_dict['p(y|x,phi) nll loss'] = pyx_nll_g * self.configs.loss.py_weight
        self.loss_dict['prior cd loss'] = prior_cd_loss * self.configs.loss.prior_e_weight

        # task loss
        meta_loss, loss_discriptor = self.validate_loss()
        self.iteration += 1
        return meta_loss , loss_discriptor, query_mse_g

    def validate_loss(self):
        task_loss = torch.stack(list(self.loss_dict.values()))
        values = task_loss.detach().tolist()
        keys = list(self.loss_dict.keys())
        self.loss_dict.clear()
        return task_loss.sum(), (keys, values)

    def sample_latent_from_q(self, x, y, num_phi_samples):
        # forward pass to compute context encode distribution for each tasks
        phi_mean, phi_std = self.encoder(x, y)
        # for each phi distribution per task, sample num_context_samples phi instances
        phi_samples = phi_mean.unsqueeze(0).repeat(num_phi_samples, 1)
        if not self.configs.model.deterministic_phi:
            phi_samples = phi_samples + phi_std.unsqueeze(0).expand(num_phi_samples, -1) * torch.randn_like(
                phi_samples)
        return phi_samples, phi_mean, phi_std

    def sample_latent_from_prior(self, phi_samples, return_summary=False):
        '''
        perform n step sgld on ebm prior model for computing learning gradient w.r.t. KL(q(phi|x)||ebm(phi)),
        phi_0 are initialized at random gaussian noise OR from buffer
        :param num_samples:
        :return:
        '''
        ak = self.configs.sgld.prior.step
        T = self.configs.sgld.prior.T
        niter = self.configs.sgld.prior.niter
        sgld_init = self.configs.sgld.prior.init

        num_samples = phi_samples.shape[0]

        for p in self.ebmprior.parameters():
            p.requires_grad = False
        # sample from p_0
        if sgld_init == 'buffer':
            buffer_idx = torch.multinomial(torch.ones(10000), num_samples, replacement=False)
            if torch.rand(1) > 0.05:
                phi_samples = self.prior_buffer[buffer_idx]
            else:
                phi_samples = torch.randn(size=(num_samples, self.configs.model.latent_dim),
                                          device='cuda')
        elif sgld_init == 'noise':
            phi_samples = torch.randn_like(phi_samples)
        elif sgld_init == 'CD':
            phi_samples = phi_samples.detach() + torch.randn_like(phi_samples) * 0.3
            phi_samples = phi_samples.cuda()
        else:
            raise ValueError

        assert phi_samples.grad is None
        # niter step sgld
        accumulated_grad = torch.tensor(0.0).cuda()
        for i in range(niter):
            phi_samples.requires_grad_(True)
            e = self.ebmprior(phi_samples)
            score = torch.autograd.grad(e.sum(), phi_samples)[0]
            with torch.no_grad():
                phi_samples = phi_samples - ak * score + 0.005 * torch.randn_like(phi_samples)
                # phi_samples -= ak * (score + 1./phi_samples) + 0.01 * torch.randn_like(phi_samples)
                accumulated_grad += torch.norm(score, dim=-1).mean()
        # replace buffer with new samples
        if sgld_init == 'buffer':
            self.prior_buffer[buffer_idx] = phi_samples.detach()

        for p in self.ebmprior.parameters():
            p.requires_grad = True
        if return_summary:
            return phi_samples.detach(), {'phi sum steps grad norm': accumulated_grad.detach()}
        else:
            return phi_samples.detach()

    def sample_xy_from_prior(self, x, y, return_summary=False):
        # constants
        eta = self.configs.sgld.prior.eta
        ak = self.configs.sgld.prior.step
        sample_x = self.configs.sgld.prior.sample_x
        # disable gradient of classifier parameters
        for p in self.ebmprior.parameters():
            p.requires_grad = False
        # negative samples per each positive sample
        # initialize samples from noise
        x = x.repeat(self.configs.sgld.prior.over_sample_ratio, 1)
        y = y.repeat(self.configs.sgld.prior.over_sample_ratio, 1)
        if sample_x:
            x = torch.randn_like(x)*0.25 + x
        else:
            x = x.detach().clone()
        y = torch.randn_like(y) + y.detach()
        # SGLD
        accumulated_gradx = torch.tensor(0.0).cuda()
        accumulated_grady = torch.tensor(0.0).cuda()
        for k in range(self.configs.sgld.prior.niter):
            x.requires_grad = True
            y.requires_grad = True
            # e shape : num_samples, batch_size, task_dim, 1
            phi, _, _ = self.sample_latent_from_q(x, y, num_phi_samples=self.configs.model.num_phi_samples)
            e_phi = self.ebmprior(phi).sum()
            # gradient is averaged over phi samples for each task
            scorex, scorey = torch.autograd.grad(e_phi, [x, y])
            with torch.no_grad():
                if sample_x: x = x - ak * scorex + eta * torch.randn_like(x)
                y = y - ak * scorey + eta * torch.randn_like(y)
                accumulated_gradx += torch.norm(scorex, dim=-1).mean()
                accumulated_grady += torch.norm(scorey, dim=-1).mean()
            x = x.detach()
            y = y.detach()
        # re-enable training:
        for p in self.ebmprior.parameters():
            p.requires_grad = True

        if return_summary:
            return x.detach(), y.detach(), {'prior x sum steps grad norm': accumulated_gradx.detach(),
                                            'prior y sum steps grad norm': accumulated_grady.detach()}

        else:
            return x.detach(), y.detach()

    def sample_xy_from_posterior(self,x,y,z, niter, return_summary=False, sample_x =True, sample_y=True, py0_std=None):
        ak = self.configs.sgld.decoder.step
        eta = self.configs.sgld.decoder.eta
        for p in self.decoder_px.parameters():
            p.requires_grad = False

        x = x.repeat(self.configs.sgld.decoder.over_sample_ratio, 1)
        x = x.detach()
        if sample_x:
            x = torch.randn_like(x) * 0.25 + x
        y = y.repeat(self.configs.sgld.decoder.over_sample_ratio, 1)
        y = y.detach()
        if sample_y and py0_std is not None:
            noise_std = py0_std.repeat(self.configs.sgld.decoder.over_sample_ratio,1).detach()
            y = torch.randn_like(y) * noise_std + y

        accumulated_gradx = torch.tensor(0.0).cuda()
        accumulated_grady = torch.tensor(0.0).cuda()
        for k in range(niter):
            x.requires_grad = True
            y.requires_grad = True
            e_x = self.decoder_px(x,y,z.detach()).sum()
            if k == 0 :
                e_x_start = e_x.detach()
            # gradient is averaged over phi samples for each task
            scorex,scorey  = torch.autograd.grad(e_x, [x,y])
            with torch.no_grad():
                if sample_x:
                    x = x - ak * scorex + eta * torch.randn_like(x)
                    accumulated_gradx += torch.norm(scorex, dim=-1).mean()
                if sample_y:
                    y = y - ak * scorey + eta * torch.randn_like(y)
                    accumulated_grady += torch.norm(scorey, dim=-1).mean()
                    y = torch.clamp(y, min=0.,max=12.0)
        e_x_final = e_x.detach()
        # re-enable training:
        for p in self.decoder_px.parameters():
            p.requires_grad = True

        if return_summary:
            # sampling = ''.join(np.array(['x','y'])[sample_x,sample_y].tolist())
            if sample_x and sample_y:
                given =""
            else:
                given ='|x' if sample_y else '|y'
            summary = {}
            if sample_x :
                summary.update({f'decoder x{given} sum steps grad norm' :accumulated_gradx.detach()})
                summary.update({f'decoder x{given} delta energy': e_x_final - e_x_start})
            if sample_y:
                summary.update({f'decoder y{given} sum steps grad norm': accumulated_grady.detach()})
                summary.update({f'decoder y{given} delta energy': e_x_final - e_x_start})
            return x.detach(),y.detach(), summary
        else:
            return x.detach(),y.detach(),

    def predict_yq(self, x_s, y_s, x_q, num_phi_samples, niter):
        '''
        :param x_s: context  x
        :param y_s: context  y
        :param x_q: given index x whose corresponding y should be estimated
        :param num_phi_samples: num of latent samples used for MC estimation of expected energy
        :param niter: number of minimization steps
        :return:
        '''
        with torch.no_grad():
            phi_s, _, _ = self.sample_latent_from_q(x_s, y_s, num_phi_samples)
        ak = self.configs.sgld.decoder.step
        T = self.configs.sgld.decoder.T
        # sample y_0
        query_size, sample_dim = x_q.shape
        if self.configs.model.gaussian_decoder:
            y_q_pred = torch.randn_like(x_q).cuda() * 0.05
            e_y, y_q_pred = self.decoder_py(x_q, y_q_pred, phi_s.detach())
        else:
            raise Exception
        assert y_q_pred.dim()==3
        return y_q_pred.detach().mean(0)

    def fill_latent_buffer(self, batch):
        support, query, task_info = batch
        x_s, y_s = [_.cuda().float().detach() for _ in support]
        self.id_phi_samples, q_s_mean, q_s_std = self.sample_latent_from_q(x_s, y_s, 1)
        self.id_phi_samples = self.id_phi_samples.view(-1, 1, self.configs.model.latent_dim).detach().cpu()
        self.ebm_phi_samples = self.sample_latent_from_prior(self.id_phi_samples).detach().cpu()
        self.gaussian_phi_samples = torch.randn_like(self.ebm_phi_samples).cpu()
        assert self.ebm_phi_samples.shape == self.id_phi_samples.shape

    def evaluate_mse(self, niter, num_phi_samples, x_q, x_s, y_q, y_s) -> np.ndarray:
        y_q_pred = self.predict_yq(x_s, y_s, x_q, num_phi_samples, niter)
        y_q_pred = y_q_pred[:, 0:1]
        # print(y_q_pred.shape, y_q.shape)
        mse = torch.nn.MSELoss(reduction='mean')
        mse_loss = mse(y_q_pred, y_q).detach().cpu().numpy()
        return mse_loss

    def evaluate_task(self, x_s,y_s,x_q,y_q, num_phi_samples, niter) -> dict:
        '''
        compute mse and expected energy of a batch of tasks
        '''
        #####################
        '''evaluate mse '''
        #####################
        task_results= {}
        phi_s, _, _ = self.sample_latent_from_q(x_s, y_s, num_phi_samples)
        y_q_pred = torch.randn_like(x_q).cuda() * 0.05
        _, y_q_pred = self.decoder_py(x_q, y_q_pred, phi_s.detach())
        y_q_mean = y_q_pred[:, : ,0:1].mean(0)
        assert y_q_mean.shape == y_q.shape
        pearson_r2 = scipy.stats.pearsonr(y_q_mean.detach().cpu().numpy().flatten(), y_q.detach().cpu().numpy().flatten())[0] ** 2
        task_results['R2'] = pearson_r2 if pearson_r2 else 0.0
        if self.configs.model.pxy_decoder and self.configs.model.refine_gaussian:
            _,y_q_pred_refined = self.sample_xy_from_posterior(x_q,y_q_mean,phi_s,
                                          niter=self.configs.model.refine_gaussian,
                                          sample_y=True,
                                          sample_x=False)
            y_q_pred_refined =  y_q_pred_refined.reshape(self.configs.sgld.decoder.over_sample_ratio, len(y_q),1).mean(0)
            pearson_r2_refined = \
            scipy.stats.pearsonr(y_q_pred_refined.detach().cpu().numpy().flatten(), y_q.detach().cpu().numpy().flatten())[
                0] ** 2
            task_results['R2 Refined'] = pearson_r2_refined if pearson_r2_refined else 0.0
        #########################
        '''evaluate task energy'''
        #########################
        with torch.no_grad():
            e_y_decoder, y_s_pred = self.decoder_py(x_s, y_s, phi_s)
            e_y_decoder = e_y_decoder.mean()
            y_s_pred = y_s_pred.mean(0)
            task_results['p(y|x phi) std'] = y_s_pred[:, 1:2].exp().mean().item()
            # data-level only
            task_results['p(y|x phi) NLL'] = e_y_decoder.item()
            if self.configs.model.px_decoder or self.configs.model.pxy_decoder:
                e_x_decoder = self.decoder_px(x_s, y_s, phi_s).mean()
                task_results['p(x|phi) NLL'] = e_x_decoder.item()
                e_xy_decoder = e_x_decoder # if pxy_decoder, then e_x_decoder:= energy for logp(x,y) the joint distribution
                if not self.configs.model.pxy_decoder: # if not direcly modelling p(x,y) using ebm decoder, then  logp(x,y) = logp(y|x) + logp(x)
                    e_xy_decoder += e_y_decoder  # so we add them up
                task_results['p(x,y) NLL'] = e_xy_decoder.item()
            else:
                e_xy_decoder = e_y_decoder
            e_prior = self.ebmprior(phi_s)
            e_prior = e_prior.mean()
            assert e_prior.shape == e_xy_decoder.shape, f'e_prior:{e_prior.shape},e_s_decoder:{e_xy_decoder.shape} '
            task_results['Energy Sum'] = (e_prior*.1 + e_xy_decoder).item()
        return task_results
