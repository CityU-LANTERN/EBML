import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.distributions import Beta


class Conv4(nn.Module):
    def __init__(self, configs):
        super(Conv4, self).__init__()
        self.configs = configs
        self.device = configs.device
        x_dim = 3
        hid_dim = configs.model.conv4.num_filters
        z_dim = configs.model.conv4.num_filters
        # sn = nn.utils.spectral_norm

        self.conv0 = nn.Conv2d(x_dim, hid_dim, 3, padding=1)
        self.conv0_post = nn.Sequential(nn.BatchNorm2d(hid_dim,track_running_stats=False),nn.ReLU(),nn.MaxPool2d(2))
        self.conv1 = self.conv_block(hid_dim, hid_dim)
        self.conv2 = self.conv_block(hid_dim, hid_dim)
        self.conv3 =  self.conv_block(hid_dim, z_dim)
        self.hid_dim = hid_dim
        self.expand_dim = configs.model.num_phi_samples
        self.train()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels,track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def functional_conv_block(self, x, weights, biases,
                              bn_weights, bn_biases, is_training):

        x = F.conv2d(x, weights, biases, padding=1)
        x = F.batch_norm(x, running_mean=None, running_var=None, weight=bn_weights, bias=bn_biases,
                         training=is_training)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x

    def forward(self, x, film_para = None):
        x = self.conv0(x)
        self.layer_one_out = x
        self.layer_one_out.requires_grad_()
        self.layer_one_out.retain_grad()
        x = self.conv0_post(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape(x.size(0), -1)
        x = x.unsqueeze(0).expand(self.expand_dim,-1,-1)
        return x

    def forward_to_N(self,x,N=3):
        x = self.conv0(x)
        x = self.conv0_post(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def forward_from_N(self,x,N=3):
        x = self.conv3(x)
        x = x.reshape(x.size(0), -1)
        x = x.unsqueeze(0).expand(self.expand_dim, -1, -1)
        return x


    def functional_forward(self, x, weights, is_training=True):
        x = F.conv2d(x, weights[f'conv0.weight'], weights[f'conv0.bias'], padding=1)
        x = F.batch_norm(x, running_mean=None,
                         running_var=None,
                         weight=weights.get(f'conv0_post.0.weight'),
                         bias=weights.get(f'conv0_post.0.bias'),
                         training=is_training)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        for block in range(1,4):
            x = self.functional_conv_block(x, weights[f'conv{block}.0.weight'], weights[f'conv{block}.0.bias'],
                                           weights.get(f'conv{block}.1.weight'), weights.get(f'conv{block}.1.bias'),
                                           is_training)

        x = x.reshape(x.size(0), -1)
        x = x.unsqueeze(0).expand(self.expand_dim, -1, -1)
        return x


