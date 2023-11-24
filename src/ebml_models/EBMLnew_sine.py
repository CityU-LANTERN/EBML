from src.utils import *
import torch
import torch.nn as nn
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm,trange
from copy import deepcopy
from src.utils import set_random_seed
matplotlib.use("Agg")

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
            sn(nn.Linear(2, self.hid_dim)),
            nn.ReLU(),
            sn(nn.Linear(self.hid_dim, self.hid_dim)),
            nn.ReLU(),
            sn(nn.Linear(self.hid_dim, self.hid_dim)),
            nn.ReLU(),
            sn(nn.Linear(self.hid_dim, self.hid_dim)),
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
        latent = self.shared_encode(xy).mean([1])
        # mu and sigma
        latent_mean = self.mu(latent)
        latent_std = self.log_sigma(latent).exp()
        return latent_mean, latent_std


class Decoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.hid_dim = configs.model.hid_dim
        self.latent_dim = configs.model.latent_dim
        sn = nn.utils.parametrizations.spectral_norm if configs.model.decoder_sn else lambda x: x
        act_fun = nn.ReLU
        # self.fcx = nn.Sequential(sn(nn.Linear(1, self.hid_dim)), act_fun())
        # self.fcxz = nn.Sequential(sn(nn.Linear(self.hid_dim + self.latent_dim, self.hid_dim)), act_fun())
        # self.fcy = nn.Sequential(sn(nn.Linear(1, self.hid_dim)), act_fun())
        # self.fc_energy = nn.Sequential(
        #     sn(nn.Linear(2 * self.hid_dim, self.hid_dim)), act_fun(),
        #     sn(nn.Linear(self.hid_dim, 1))
        # )
        self.fc_energy = nn.Sequential(
            sn(nn.Linear(1 * self.latent_dim + 2, self.hid_dim)), act_fun(),
            sn(nn.Linear(self.hid_dim, self.hid_dim)), act_fun(),
            sn(nn.Linear(self.hid_dim, self.hid_dim)), act_fun(),
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
        # shape (num_phi_samples, batch_size, task_dim ,latent_dim)
        x = x.unsqueeze(0).expand(num_phi_samples, -1, -1, -1)
        y = y.unsqueeze(0).expand(num_phi_samples, -1, -1, -1)
        z = phi.unsqueeze(-2).expand(-1, -1, x.shape[-2], -1)
        # concat [x,z] -> conditional features
        # x = self.fcx(x)
        # xz = self.fcxz(torch.cat([x, z], dim=-1))
        # y = self.fcy(y)
        e_xyz = self.fc_energy(torch.cat([y, x, z], dim=-1))  # shape ( batch_size, task_dim ,1)
        return e_xyz, None


class Prior(nn.Module):
    def __init__(self, configs):
        super(Prior, self).__init__()
        sn = nn.utils.parametrizations.spectral_norm if configs.model.prior_sn else lambda x: x
        self.prior_weight = configs.loss.kl_weight
        self.use_prior = not configs.model.deterministic_phi
        self.fc = nn.Sequential(
            nn.Linear(configs.model.latent_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1)
        )

    def forward(self, phi):
        e = self.fc(phi)
        if self.use_prior:
            base_e = torch.linalg.norm(phi,dim=-1, keepdim=True) / phi.shape[-1]
            e = e+base_e * self.prior_weight
        return e


class GaussianDecoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.hid_dim = configs.model.hid_dim
        self.latent_dim = configs.model.latent_dim
        self.input_dim = 1
        self.mlp_energy = nn.Sequential(
            nn.Linear(2 + self.latent_dim, self.hid_dim),
            nn.SiLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.SiLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.SiLU(),
            nn.Linear(self.hid_dim,1)
        )
        self.mlp_gaussian = nn.Sequential(
            nn.Linear(1 + self.latent_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim,2)
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
        y_pred = self.mlp_gaussian(xz)
        e_xy = self.mlp_energy(torch.cat([xz,y],dim=-1))
        nll_xy = 0.5 * ((y[:, :, :, 0:1] - y_pred[:, :, :, 0:1]) / y_pred[:, :, :, 1:2].exp()).square() + y_pred[:, :, :, 1:2]
        e_xy = nll_xy.detach() * .0 + e_xy # use gaussian-like energy as the base energy
        return e_xy, nll_xy, y_pred


class EBMMetaRegressor(nn.Module):
    def __init__(self, configs):
        super(EBMMetaRegressor, self).__init__()
        self.configs = configs
        self.encoder = Encoder(configs)
        self.decoder = GaussianDecoder(configs) if self.configs.model.gaussian_decoder else Decoder(configs)
        self.ebmprior = Prior(configs)
        self.device = configs.device
        self.iteration = 0
        # projection head for latent variable
        self.projection_head = nn.Linear(configs.model.latent_dim, 32).cuda()
        self.con_loss = SupConLoss(temperature=0.5)
        # optimizers
        self.optimizer = torch.optim.Adam([{'params': self.encoder.parameters()},
                                        {'params': self.decoder.parameters()},
                                        {'params': self.projection_head.parameters()},
                                          {'params': self.ebmprior.parameters()}],
                                              lr=configs.optimizer.lr, betas=(configs.optimizer.beta1, configs.optimizer.beta2))
        lambda_en = lambda epoch: 1.0 ** (epoch // 10)
        lambda_constant = lambda epoch: 1.0
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                           lr_lambda=[lambda_constant, lambda_constant, lambda_constant,lambda_constant],
                                                           verbose=False)
        # sgld buffers
        self.prior_buffer = torch.randn(10000, self.configs.model.latent_dim, dtype=torch.float32).cuda()
        self.loss_dict = {}

    def meta_update(self, batch):
        """
        compute training batch loss and update learner parameters
        """
        ####################
        '''features processing'''
        ####################
        supp_batch, query_batch, title_batch = batch
        x_s, y_s = [_.cuda().float() for _ in supp_batch]
        x_q, y_q = [_.cuda().float() for _ in query_batch]
        x_sq = torch.cat([x_s, x_q], dim=1)
        y_sq = torch.cat([y_s, y_q], dim=1)
        if self.configs.dataset.varying_shot:
            n_support = np.random.choice([2,3,4,5])
            x_s = x_s[:, :n_support]
            y_s = y_s[:, :n_support]
        #####################
        '''ebm prior loss'''
        #####################
        # sample posterior phi from amortized encoder
        phi_s_batch, q_s_mean, q_s_std = self.sample_latent_from_q(x_s, y_s, self.configs.model.num_phi_samples)
        # sample prior phi from ebm prior model
        if not self.configs.sgld.prior.sample_in_xy:
            ############### sample in phi space ###############
            phi_s_ebm, prior_sgld_summary = self.sample_latent_from_prior(phi_s_batch, return_summary=True)
        else:
            ############### sample in xy space ###############
            x_, y_ ,prior_sgld_summary= self.sample_xy_from_prior(x_s, y_s, return_summary=True)
            phi_s_ebm, _, _ = self.sample_latent_from_q(x_, y_, self.configs.model.num_phi_samples)
        # compute learning loss wrt ebm prior
        ephi_post = self.ebmprior(phi_s_batch)
        ephi_prior = self.ebmprior(phi_s_ebm)  # shape : (num_samples,1)
        # print(ephi_post==ephi_prior)
        prior_cd_loss = ephi_post.mean() - ephi_prior.mean()
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
            pass
            #self.loss_dict['l2 phi'] = torch.norm(phi_s_batch, dim=-1, p=2).mean() * 0.01
        else:
            kl_loss = gaussian_kl(q_s_mean,
                                  q_s_std ** 2,
                                  torch.zeros_like(q_s_mean),
                                  torch.ones_like(q_s_std)).mean()
            self.loss_dict['kl loss'] = kl_loss * self.configs.loss.kl_weight

        if False:
            # positive energy
            _, nll, _ = self.decoder(x_sq, y_sq, phi_s_batch)
        else:
            epos, nll, _ = self.decoder(x_sq, y_sq, phi_s_batch)
            # sample a batch of negative tasks
            x_neg, y_neg, xy_sgld_summary = self.sample_tasks_from_posterior(x_sq, y_sq, phi_s_batch,
                                                            sample_x=self.configs.sgld.decoder.sample_x,
                                                            return_summary=True)
            eneg, _,  _ = self.decoder(x_neg, y_neg, phi_s_batch)
            vae_cd_loss = epos.mean() - eneg.mean()
            self.loss_dict.update(xy_sgld_summary)
        self.loss_dict["p(y|x) nll"] = nll.mean([0, 1]).sum()
        assert not torch.isnan(vae_cd_loss)
        ################
        '''l2 energy'''
        ################
        if self.configs.loss.l2_weight > 0:
            self.loss_dict['l2 energy'] =  (ephi_post.square().mean() + ephi_prior.square().mean())*1.0
            self.loss_dict['e prior pos'] = ephi_post.detach().mean()
            self.loss_dict['e prior neg'] = ephi_prior.detach().mean()
            if True:
                self.loss_dict['l2 energy'] +=(epos.square().mean() + eneg.square().mean())*0.1
                assert not torch.isnan(self.loss_dict['l2 energy'])
                self.loss_dict['e decoder pos'] = epos.detach().mean()
                self.loss_dict['e decoder neg'] = eneg.detach().mean()
        ################
        '''meta-loss'''
        ################
        self.loss_dict['decoder cd loss'] = vae_cd_loss * 0.1
        self.loss_dict['prior cd loss'] = prior_cd_loss * self.configs.loss.ebmprior_weight

        # task loss
        meta_loss, loss_discriptor = self.validate_loss()
        self.optimizer.zero_grad()
        meta_loss.backward()
        # gradient_clip :
        self.optimizer.step()
        self.scheduler.step()
        self.iteration += 1
        return loss_discriptor

    def validate_loss(self):
        task_loss = torch.stack(list(self.loss_dict.values()))
        values = task_loss.detach().tolist()
        keys = list(self.loss_dict.keys())
        self.loss_dict.clear()
        return task_loss.sum(), (keys, values)

    def evaluate_prior_energy(self, latent_representation):
        # ebm prior energy
        e_prior_f = self.ebmprior(latent_representation)  # shape: (num_phi_samples, batch_size, 1)
        e_prior_p0 = 0.5 * latent_representation.square().sum(dim=-1,
                                                              keepdim=True)  # shape: (num_phi_samples, batch_size, 1)
        assert e_prior_p0.shape ==e_prior_f.shape
        return e_prior_f+e_prior_p0

    def sample_latent_from_q(self, x, y, num_phi_samples):
        if x.dim() < 3:
            raise Exception('input_dim should match (batch_size, num_samples, sample_size)')
        # forward pass to compute context encode distribution for each tasks
        phi_mean, phi_std = self.encoder(x, y)
        # for each phi distribution per task, sample num_context_samples phi instances
        phi_samples = phi_mean.unsqueeze(0).repeat(num_phi_samples, 1, 1)
        if not self.configs.model.deterministic_phi:
            phi_samples = phi_samples + phi_std.unsqueeze(0).expand(num_phi_samples, -1, -1) * torch.randn_like(
                phi_samples)
        return phi_samples, phi_mean, phi_std

    def sample_latent_from_prior(self, phi_samples,return_summary=False):
        '''
        perform n step sgld on ebm prior model for computing learning gradient w.r.t. KL(q(phi|x)||ebm(phi)),
        phi_0 are initialized at random gaussian noise OR from buffer
        :param num_samples:
        :return:
        '''
        ak = self.configs.sgld.prior.step
        T = self.configs.sgld.prior.T
        niter=self.configs.sgld.prior.niter
        sgld_init = self.configs.sgld.prior.init
        n_phis, batch,dim = phi_samples.shape
        num_samples = n_phis * batch

        for p in self.ebmprior.parameters():
            p.requires_grad = False
        # sample from p_0
        if sgld_init == 'buffer':
            buffer_idx = torch.multinomial(torch.ones(10000), num_samples, replacement=False)
            use_noise = torch.less(torch.rand(num_samples, device=phi_samples.device),0.05).float()[:,None]
            buffer_samples = self.prior_buffer[buffer_idx].detach()
            phi_samples = (1-use_noise) * buffer_samples + use_noise* torch.randn_like(buffer_samples)
            phi_samples = phi_samples.reshape(n_phis, batch, dim)
        elif sgld_init =='noise':
            phi_samples = torch.randn_like(phi_samples)
        elif sgld_init =='CD':
            phi_samples = phi_samples.detach() + torch.randn_like(phi_samples) * 0.3
            phi_samples = phi_samples.cuda()
        else:
            raise ValueError

        assert phi_samples.grad is None

        accumulated_grad = torch.tensor(0.0).cuda()
        for i in range(niter):
            phi_samples.requires_grad_(True)
            e = self.ebmprior(phi_samples)
            score = torch.autograd.grad(e.sum(), phi_samples)[0]
            with torch.no_grad():
                phi_samples = phi_samples - ak * score +  0.005 * torch.randn_like(phi_samples)
                accumulated_grad += torch.norm(score, dim=-1).mean()

        if sgld_init == 'buffer':
            self.prior_buffer[buffer_idx] = phi_samples.detach().reshape(num_samples,dim)

        for p in self.ebmprior.parameters():
            p.requires_grad = True
        if return_summary:
            return phi_samples.detach(), {'phi sum steps grad norm' : accumulated_grad.detach()}
        else:
            return phi_samples.detach()

    def sample_xy_from_prior(self, x,y,return_summary=False):
        # constants
        ak = self.configs.sgld.prior.step
        sample_x = self.configs.sgld.prior.sample_x
        eta = self.configs.sgld.prior.eta
        clip_grad = self.configs.sgld.prior.clip_grad
        # disable gradient of classifier parameters
        for p in self.ebmprior.parameters():
            p.requires_grad = False
        # initialize samples from noise
        x = x.repeat(self.configs.sgld.prior.over_sample_ratio,1,1)
        y = y.repeat(self.configs.sgld.prior.over_sample_ratio,1,1)
        if sample_x:
            x = torch.rand_like(x) * 10 - 5.0
        else:
            x = x.detach().clone()
        y = torch.randn_like(y)
        # SGLD
        accumulated_gradx = torch.tensor(0.0).cuda()
        accumulated_grady = torch.tensor(0.0).cuda()
        for k in range(1, self.configs.sgld.prior.niter + 1):
            x.requires_grad = True
            y.requires_grad = True
            # e shape : num_samples, batch_size, task_dim, 1
            phi,_,_ = self.sample_latent_from_q(x,y,num_phi_samples = self.configs.model.num_phi_samples)
            e_phi = self.ebmprior(phi).sum()
            # gradient is averaged over phi samples for each task
            scorex, scorey = torch.autograd.grad(e_phi, [x, y])
            with torch.no_grad():
                if clip_grad > 0:
                    scorey = torch.clamp(scorey, min=-clip_grad, max=clip_grad)
                    scorex = torch.clamp(scorex, min=-clip_grad, max=clip_grad)
                if sample_x: x = x - ak * scorex + eta * torch.randn_like(x)  #todo this was 0.005
                y = y - ak * scorey + eta * torch.randn_like(y)
                y = torch.clamp(y, min=-5, max=5)
                x = torch.clamp(x, min=-5, max=5)
                accumulated_gradx += torch.norm(scorex, dim=-1).mean()
                accumulated_grady += torch.norm(scorey, dim=-1).mean()
            x = x.detach()
            y = y.detach()
        # re-enable training:
        for p in self.ebmprior.parameters():
            p.requires_grad = True

        if return_summary:
            return x.detach(), y.detach(), {'prior x sum steps grad norm' : accumulated_gradx.detach(),
                                            'prior y sum steps grad norm' : accumulated_grady.detach()}

        else:
            return x.detach(), y.detach()

    def sample_tasks_from_posterior(self, x, y, z, sample_x=False, return_summary=False):
        """
        Sample the same number of negative tasks = cardinality of input tasks (x_init,y_init),
        from classifier distribution given latent variable phi  P( x , y | phi~q(phi):=f(x,y) ).
        """
        # constants
        T = self.configs.sgld.decoder.T
        ak = self.configs.sgld.decoder.step
        sgld_init = self.configs.sgld.decoder.init
        eta = self.configs.sgld.decoder.eta
        clip_grad = self.configs.sgld.decoder.clip_grad
        for p in self.decoder.parameters():
            p.requires_grad = False
        # initialize samples from noise
        x = x.repeat(1,16,1)
        y = y.repeat(1,16,1)
        if sgld_init == 'noise':
            if sample_x:
                x = torch.rand_like(x) * 13 - 6.5
            y = torch.rand_like(y) * 10 - 5
        elif sgld_init == 'CD':
            if sample_x:
                x = x.detach()+ torch.randn_like(x) * 0.25
            y = y.detach() + torch.randn_like(y) * 0.25
        else:
            raise ValueError
        x = x.detach()
        accumulated_gradx =  torch.tensor(0.0).cuda()
        accumulated_grady =  torch.tensor(0.0).cuda()
        # SGLD
        for k in range(self.configs.sgld.decoder.niter):
            x.requires_grad = True
            y.requires_grad = True
            # e shape : num_samples, batch_size, task_dim, 1
            e_xy,_, _ = self.decoder(x, y, z.detach())
            # gradient is averaged over phi samples for each task
            scorex, scorey = torch.autograd.grad(e_xy.sum(), [x, y])
            with torch.no_grad():
                if clip_grad > 0:
                    scorey = torch.clamp(scorey,min=-clip_grad, max=clip_grad)
                    scorex = torch.clamp(scorex, min=-clip_grad, max=clip_grad)
                if sample_x:  x -= ak * scorex + eta * torch.randn_like(x)
                y -= ak * scorey + eta * torch.randn_like(x)
                y = torch.clamp(y, min=-6.5, max=6.5)
                x = torch.clamp(x, min=-5.5, max=5.5)
                accumulated_gradx += torch.norm(scorex, dim=-1).mean()
                accumulated_grady += torch.norm(scorey, dim=-1).mean()

        for p in self.decoder.parameters():
            p.requires_grad = True
        if return_summary:
            return x.detach(), y.detach(), {'x sum steps grad norm' : accumulated_gradx.detach(),
                                            'y sum steps grad norm' : accumulated_grady.detach()}
        else:
            return x.detach(), y.detach()

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
        # sgld params:
        ak = self.configs.sgld.decoder.step
        T = self.configs.sgld.decoder.T
        # sample y_0
        batch_size, query_size, sample_dim = x_q.shape

        if self.configs.model.gaussian_decoder:
            y_q_pred = torch.randn_like(x_q).cuda() * 0.05
            _,_, y_q_pred = self.decoder(x_q, y_q_pred, phi_s.detach())
        else:
            _n_predictions = 4
            x_q = x_q.repeat(1, 1, _n_predictions).view(batch_size,-1,sample_dim)
            y_q_pred = torch.randn_like(x_q).cuda() * 0.05
            # for loop for sampling y
            for j in range(niter):
                y_q_pred.requires_grad = True
                e_y,_,y_q_pred = self.decoder(x_q, y_q_pred, phi_s.detach())
                # gradient is averaged over phi samples for each task
                scorey = torch.autograd.grad(e_y.sum(), y_q_pred)[0]
                with torch.no_grad():
                    y_q_pred -= ak * scorey + (2. * ak * T) ** 0.5 * torch.randn_like(y_q_pred)
            y_q_pred = y_q_pred.view(1, batch_size, query_size, _n_predictions).mean(-1,keepdim=True)
        return y_q_pred.detach().mean(0)

    def fill_latent_buffer(self,batch):
        support, query, task_info = batch
        x_s, y_s = [_.cuda().float().detach() for _ in support]
        self.id_phi_samples, q_s_mean, q_s_std = self.sample_latent_from_q(x_s, y_s, 1)
        self.id_phi_samples = self.id_phi_samples.view(-1,1,self.configs.model.latent_dim)
        self.ebm_phi_samples = self.sample_latent_from_prior(self.id_phi_samples).detach().cpu()
        self.gaussian_phi_samples  = torch.randn_like(self.ebm_phi_samples).cpu()
        self.id_phi_samples = self.id_phi_samples.detach().cpu()
        assert self.ebm_phi_samples.shape == self.id_phi_samples.shape

    def evaluate_mse(self, niter, num_phi_samples, x_q, x_s, y_q, y_s) -> np.ndarray:
        y_q_pred = self.predict_yq(x_s, y_s, x_q, num_phi_samples, niter)
        y_q_pred = y_q_pred[:,:,0:1]
        # print(y_q_pred.shape, y_q.shape)
        mse = torch.nn.MSELoss(reduction='none')
        mse_loss = torch.mean(mse(y_q_pred, y_q), dim=(1, 2)).detach().cpu().numpy()
        return mse_loss

    def evaluate_batch(self, batch, num_phi_samples, niter) -> [np.ndarray,np.ndarray]:
        '''
        compute mse and expected energy of a batch of tasks
        '''
        support, query, task_info = batch
        x_s, y_s = [_.cuda().float().detach() for _ in support]
        x_q, y_q = [_.cuda().float().detach() for _ in query]
        #####################
        '''evaluate mse '''
        #####################
        batch_results = {}
        mse_loss = self.evaluate_mse(niter, num_phi_samples, x_q=x_q, x_s=x_s, y_q=y_q, y_s=y_s)
        batch_results["MSE query"] = mse_loss
        #########################
        '''evaluate task energy'''
        #########################
        # ood score = un-normalized support energy
        # i.e., log p(x,y) proportional to ebm_decoder(x,y|phi) + ebm_prior(phi)
        phi_q_samples, q_s_mean, q_s_std = self.sample_latent_from_q(x_s, y_s, num_phi_samples)
        with torch.no_grad():
            # energy sum
            e_s_decoder,gauss_nll,_ = self.decoder(x_s, y_s, phi_q_samples)
            e_s_decoder = e_s_decoder.mean([0, -1, -2])
            gauss_nll = gauss_nll.mean([0, -1, -2])
            batch_results["ebm SNLL"] = e_s_decoder.detach().cpu().numpy()
            e_prior = self.ebmprior(phi_q_samples)
            e_prior = e_prior.mean([0,-1])  # num_phi, bs, 1
            assert e_prior.shape ==e_s_decoder.shape ,f'e_prior:{e_prior.shape},e_s_decoder:{e_s_decoder.shape} '
            batch_results["energy sum"] = (e_prior + e_s_decoder).detach().cpu().numpy()
            self.ebm_phi_samples = self.ebm_phi_samples.expand(-1,len(x_s),-1)
            e_s_decoder, _ ,_= self.decoder(x_s, y_s, self.ebm_phi_samples.to(x_s.device))
            e_s_decoder = e_s_decoder.mean([0, -1, -2])
            batch_results["MC average"]= e_s_decoder.detach().cpu().numpy()
        # score sum
        x_s.requires_grad_(True)
        y_s.requires_grad_(True)
        e_s_decoder, _,_ = self.decoder(x_s, y_s, phi_q_samples)
        grad_x,grad_y =torch.autograd.grad(e_s_decoder.sum(),[x_s,y_s])
        phi_q_samples.detach_().requires_grad_(True)
        e_prior = self.ebmprior(phi_q_samples)
        grad_phi = torch.autograd.grad(e_prior.sum(), phi_q_samples)[0]
        grad_x,grad_y,grad_phi = torch.abs(grad_x),torch.abs(grad_y), torch.abs(grad_phi)
        grad = grad_x.mean([-1,-2]) + grad_y.mean([-1,-2]) + grad_phi.mean([0,-1])
        batch_results["score sum"] = grad.cpu().numpy()
        return batch_results

    def estimate_NLL(self, batch, num_phi_samples):
        '''
        Estimate the expected NLL -log p(y|x) over batch of tasks. Partition constants at each x are approximated by
        discrete summation.
        :param batch:
        :param num_phi_samples:
        :return:
        '''
        STEPS = 1024
        torch.manual_seed(0)
        with torch.no_grad():
            support, query, task_info = batch
            x_s, y_s = [_.cuda().float().detach() for _ in support]
            x_q, y_q = [_.cuda().float().detach() for _ in query]
            y = torch.linspace(-5.0, 5.0, steps=STEPS).cuda()
            dy = 10 / STEPS  # delta y for approximation
            NLL = []
            phi_s, _, _ = self.sample_latent_from_q(x_s, y_s, num_phi_samples)
            e_q,_ = self.decoder(x_q, y_q, phi_s).squeeze(-1)
            for i in range(x_s.shape[0]):  # iterate over tasks in a batch
                x = x_q[i].view(-1)
                grid_x, grid_y = torch.meshgrid(x, y)
                phi_i = phi_s[:, i, :].unsqueeze(1)
                e_xy_i,_ = self.decoder(grid_x.reshape(1, -1, 1),
                                      grid_y.reshape(1, -1, 1), phi_i).view(num_phi_samples, -1, STEPS)
                e_xy_i = e_xy_i.double() / self.configs.sgld.decoder.T
                e_q_i = e_q[:, i].double() / self.configs.sgld.decoder.T
                p_xy_i = (-e_xy_i).exp().sum(-1) * dy  # discrete approximation: = SUM[y'](p(y'|x) * dy)
                p_q_i = (-e_q_i).exp()
                LL_i = (p_q_i / p_xy_i).mean(0).log().mean()
                NLL.append(-LL_i.item())
            NLL = np.array(NLL)
        return NLL

    def compute_energy_over_grid(self, phi, normalized=True):
        '''
        :param phi: shape( num_phi_samples, batch_size, latent_dim)
        :return: energy over the defined xy grid
        '''
        with torch.no_grad():
            x = torch.arange(-5., 5., step=0.05).cuda()
            y = torch.arange(-5., 5., step=0.05).cuda()
            grid_x, grid_y = torch.meshgrid(x, y)
            e,_,_= self.decoder(grid_x.reshape(1, -1, 1).expand(phi.shape[1], -1, -1),
                             grid_y.reshape(1, -1, 1).expand(phi.shape[1], -1, -1), phi)
            e = e/ self.configs.sgld.decoder.T
            if normalized:
                Z = torch.sum(torch.exp(-e), dim=(1, 2))
                for i in range(Z.shape[0]):
                    e[i] = torch.exp(-e[i]) / Z[i]
            e = e.reshape(phi.shape[0], phi.shape[1], len(x), len(y))
            e = torch.mean(e, dim=0)
        return e.detach(), grid_x, grid_y


    def plot_posterior_energy_pdf(self, batch, num_phi_samples, savepath, niter, xmin=-5., xmax=10., ymin=-2.5, ymax=0.):
        num_plot = 4
        support, query, task_info = batch
        x_s, y_s = [_.cuda().float() for _ in support]
        x_q, y_q = [_.cuda().float() for _ in query]

        n_support = np.random.choice([5])
        x_s = x_s[:, :n_support]
        y_s = y_s[:, :n_support]

        # restrict plots to num_plot tasks
        if x_s.shape[0] > num_plot:
            x_s = x_s[0:num_plot]
            y_s = y_s[0:num_plot]
            x_q = x_q[0:num_plot]
            y_q = y_q[0:num_plot]
        '''plots arrangement'''
        fig, ax = plt.subplots(x_s.shape[0], 3, sharex=True, sharey=True, figsize=(4 * 3, 4 * x_s.shape[0]),
                               gridspec_kw={'wspace': 0.05, 'hspace': 0.25})
        task_plots = ax[:, 0]
        p_plots = ax[:, 1]
        e_plots = ax[:, 2]
        # infer task specific context ON SUPPORT SET ONLY
        with torch.no_grad():
            phi_s_batch, _, _ = self.sample_latent_from_q(x_s, y_s, num_phi_samples)

        y_qpred = self.predict_yq(x_s, y_s, x_q, num_phi_samples=num_phi_samples, niter=niter)
        y_qpred = y_qpred[:,:,0]
        for i in range(len(task_plots)):
            task_plots[i].scatter(x_q[i].detach().cpu().numpy(), y_q[i].detach().cpu().numpy(), marker='.',
                                  c='midnightblue', s=3)
            task_plots[i].scatter(x_q[i].detach().cpu().numpy(), y_qpred[i].detach().cpu().numpy(), marker='.', c='k',
                                  s=3)
            task_plots[i].plot(x_s[i].detach().cpu().numpy(), y_s[i].detach().cpu().numpy(), '.', c='dodgerblue',
                               markersize=7)

        with torch.no_grad():
            e, grid_x, grid_y = self.compute_energy_over_grid(phi_s_batch)
            _, max_indx = torch.max(e, dim=-1)
            for i in range(len(p_plots)):
                p_plots[i].imshow(np.rot90(e[i].detach().cpu().numpy()), cmap=plt.cm.viridis,
                                  extent=[xmin, xmax, ymin, ymax], aspect='auto')
                p_plots[i].plot(x_s[i].detach().cpu().numpy(), y_s[i].detach().cpu().numpy(), '.', markersize=7,
                                color='dodgerblue')


        with torch.no_grad():
            e, grid_x, grid_y = self.compute_energy_over_grid(phi_s_batch, normalized=False)
            _, max_indx = torch.max(e, dim=-1)
            for i in range(len(e_plots)):
                e_plots[i].imshow(np.rot90(e[i].detach().cpu().numpy()), cmap=plt.cm.viridis,
                                  extent=[xmin, xmax, ymin, ymax], aspect='auto')
                e_plots[i].plot(grid_x[:, 0].detach().cpu().numpy(), grid_y[0, max_indx[i]].detach().cpu().numpy(),
                                'k.', alpha=0.25)
                e_plots[i].plot(x_s[i].detach().cpu().numpy(), y_s[i].detach().cpu().numpy(), 'r.', markersize=5)
            e_plots[0].set_xlim([xmin, xmax])
            e_plots[0].set_ylim([ymin, ymax])

        for i, ax in enumerate(fig.axes):
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        plt.savefig(savepath, dpi=500)
        plt.close(fig)

