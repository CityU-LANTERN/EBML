import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import copy
from typing import Mapping, Any
########################
#  Resnet-18 Backbone  #
########################
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

########
# Resnet18
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, bn_fn, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = bn_fn(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = bn_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, bn_fn):
        super(ResNet, self).__init__()
        self.initial_pool = False
        inplanes = self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2, padding=1, bias=False)
        self.bn1 = bn_fn(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, inplanes, layers[0], bn_fn)
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], bn_fn, stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], bn_fn, stride=2)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], bn_fn, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, bn_fn, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                bn_fn(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, bn_fn, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_fn))

        return nn.Sequential(*layers)

    def forward(self, x, param_dict=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.initial_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

    def get_layer_output(self, x, param_dict, layer_to_return):
        if layer_to_return == 0:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            if self.initial_pool:
                x = self.maxpool(x)
            return x
        else:
            resnet_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            layer = layer_to_return - 1
            for block in range(self.layers[layer]):
                x = resnet_layers[layer][block](x, param_dict[layer][block]['gamma1'],
                                                param_dict[layer][block]['beta1'],
                                                param_dict[layer][block]['gamma2'], param_dict[layer][block]['beta2'])
            return x

    @property
    def output_size(self):
        return 512


class BasicBlockFilm(nn.Module):
    """
    Extension to standard ResNet block (https://arxiv.org/abs/1512.03385) with FiLM layer adaptation. After every batch
    normalization layer, we add a FiLM layer (which applies an affine transformation to each channel in the hidden
    representation). As we are adapting the feature extractor with an external adaptation network, we expect parameters
    to be passed as an argument of the forward pass.
    """
    expansion = 1

    def __init__(self, inplanes, planes, bn_fn, stride=1, downsample=None):
        super(BasicBlockFilm, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = bn_fn(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = bn_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, gamma1, beta1, gamma2, beta2):
        """
        Implements a forward pass through the FiLM adapted ResNet block. FiLM parameters for adaptation are passed
        through to the method, one gamma / beta set for each convolutional layer in the block (2 for the blocks we are
        working with).
        :param x: (torch.tensor) Batch of images to apply computation to.
        :param gamma1: (torch.tensor) Multiplicative FiLM parameter for first conv layer (one for each channel).
        :param beta1: (torch.tensor) Additive FiLM parameter for first conv layer (one for each channel).
        :param gamma2: (torch.tensor) Multiplicative FiLM parameter for second conv layer (one for each channel).
        :param beta2: (torch.tensor) Additive FiLM parameter for second conv layer (one for each channel).
        :return: (torch.tensor) Resulting representation after passing through layer.
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self._film(out, gamma1, beta1)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self._film(out, gamma2, beta2)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def _film(self, x, gamma, beta):
        '''
        :param x: B C W H, B = num_phi_samples(P) * num_query/supp(S)
        :param gamma: S C
        :param beta: S C
        :return:
        '''
        batch_size, C, W, H = x.shape
        num_phi_samples = gamma.shape[0]
        x = x.view(num_phi_samples, -1, C, W, H)
        gamma = gamma[:, None, :, None, None]
        beta = beta[:, None, :, None, None]
        x = gamma * x + beta
        x = x.view(batch_size, C, W, H)
        return x


class FilmResNet(ResNet):
    """
    Wrapper object around BasicBlockFilm that constructs a complete ResNet with FiLM layer adaptation. Inherits from
    ResNet object, and works with identical logic.
    """

    def __init__(self, block, layers, bn_fn):
        ResNet.__init__(self, block, layers, bn_fn)
        self.layers = layers

    def forward(self, x, param_dict):
        """
        Forward pass through ResNet. Same logic as standard ResNet, but expects a dictionary of FiLM parameters to be
        provided (by adaptation network objects).
        :param x: (torch.tensor) Batch of images to pass through ResNet.
        :param param_dict: (list::dict::torch.tensor) One dictionary for each block in each layer of the ResNet,
                           containing the FiLM adaptation parameters for each conv layer in the model.
        :return: (torch.tensor) Feature representation after passing through adapted network.
        """
        x = self.conv1(x)
        # for YOPO
        # self.layer_one_out = x
        # self.layer_one_out.requires_grad_()
        # self.layer_one_out.retain_grad()
        x = self.bn1(x)
        x = self.relu(x)
        if self.initial_pool:
            x = self.maxpool(x)
        # expand dim 0 to match dim 0 of gammas and betas
        num_phi_samples = param_dict[0][0]['gamma1'].shape[0]
        num_samples, C, W, H = x.shape
        # merge dim 0 num_phi_samples and dim 1 num_query/support
        x = x.unsqueeze(0).contiguous().expand(num_phi_samples, -1, -1, -1, -1).view(-1, C, W, H)
        for block in range(self.layers[0]):
            x = self.layer1[block](x, param_dict[0][block]['gamma1'], param_dict[0][block]['beta1'],
                                   param_dict[0][block]['gamma2'], param_dict[0][block]['beta2'])
        for block in range(self.layers[1]):
            x = self.layer2[block](x, param_dict[1][block]['gamma1'], param_dict[1][block]['beta1'],
                                   param_dict[1][block]['gamma2'], param_dict[1][block]['beta2'])
        for block in range(self.layers[2]):
            x = self.layer3[block](x, param_dict[2][block]['gamma1'], param_dict[2][block]['beta1'],
                                   param_dict[2][block]['gamma2'], param_dict[2][block]['beta2'])
        for block in range(self.layers[3]):
            x = self.layer4[block](x, param_dict[3][block]['gamma1'], param_dict[3][block]['beta1'],
                                   param_dict[3][block]['gamma2'], param_dict[3][block]['beta2'])

        x = self.avgpool(x)
        x = x.view(num_phi_samples, num_samples, self.output_size)
        return x


class FilmResNetOriginal(ResNet):
    """
    Wrapper object around BasicBlockFilm that constructs a complete ResNet with FiLM layer adaptation. Inherits from
    ResNet object, and works with identical logic.
    """
    def __init__(self, block, layers,bn_fn):
        ResNet.__init__(self, block, layers, bn_fn)
        self.layers = layers

    def forward(self, x, param_dict):
        """
        Forward pass through ResNet. Same logic as standard ResNet, but expects a dictionary of FiLM parameters to be
        provided (by adaptation network objects).
        :param x: (torch.tensor) Batch of images to pass through ResNet.
        :param param_dict: (list::dict::torch.tensor) One dictionary for each block in each layer of the ResNet,
                           containing the FiLM adaptation parameters for each conv layer in the model.
        :return: (torch.tensor) Feature representation after passing through adapted network.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.initial_pool:
            x = self.maxpool(x)

        for block in range(self.layers[0]):
            x = self.layer1[block](x, param_dict[0][block]['gamma1'], param_dict[0][block]['beta1'],
                                   param_dict[0][block]['gamma2'], param_dict[0][block]['beta2'])
        for block in range(self.layers[1]):
            x = self.layer2[block](x, param_dict[1][block]['gamma1'], param_dict[1][block]['beta1'],
                                   param_dict[1][block]['gamma2'], param_dict[1][block]['beta2'])
        for block in range(self.layers[2]):
            x = self.layer3[block](x, param_dict[2][block]['gamma1'], param_dict[2][block]['beta1'],
                                   param_dict[2][block]['gamma2'], param_dict[2][block]['beta2'])
        for block in range(self.layers[3]):
            x = self.layer4[block](x, param_dict[3][block]['gamma1'], param_dict[3][block]['beta1'],
                                   param_dict[3][block]['gamma2'], param_dict[3][block]['beta2'])

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


class BasicBlockFilmOriginal(nn.Module):
    """
    Extension to standard ResNet block (https://arxiv.org/abs/1512.03385) with FiLM layer adaptation. After every batch
    normalization layer, we add a FiLM layer (which applies an affine transformation to each channel in the hidden
    representation). As we are adapting the feature extractor with an external adaptation network, we expect parameters
    to be passed as an argument of the forward pass.
    """
    expansion = 1

    def __init__(self, inplanes, planes, bn_fn , stride=1, downsample=None):
        super(BasicBlockFilmOriginal, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = bn_fn(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = bn_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, gamma1, beta1, gamma2, beta2):
        """
        Implements a forward pass through the FiLM adapted ResNet block. FiLM parameters for adaptation are passed
        through to the method, one gamma / beta set for each convolutional layer in the block (2 for the blocks we are
        working with).
        :param x: (torch.tensor) Batch of images to apply computation to.
        :param gamma1: (torch.tensor) Multiplicative FiLM parameter for first conv layer (one for each channel).
        :param beta1: (torch.tensor) Additive FiLM parameter for first conv layer (one for each channel).
        :param gamma2: (torch.tensor) Multiplicative FiLM parameter for second conv layer (one for each channel).
        :param beta2: (torch.tensor) Additive FiLM parameter for second conv layer (one for each channel).
        :return: (torch.tensor) Resulting representation after passing through layer.
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self._film(out, gamma1, beta1)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self._film(out, gamma2, beta2)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def _film(self, x, gamma, beta):
        gamma = gamma[None, :, None, None]
        beta = beta[None, :, None, None]
        return gamma * x + beta


class TSAConvBlock(nn.Module):
    def __init__(self, orig_conv, configs):
        super(TSAConvBlock, self).__init__()
        # the original conv layer
        self.conv = copy.deepcopy(orig_conv)
        self.conv.weight.requires_grad = False
        planes, in_planes, _, _ = self.conv.weight.size()
        stride, _ = self.conv.stride
        # task-specific adapters
        if 'alpha' not in configs.tta.tsa.opt: #args['test.tsa_opt']:
            self.ad_type = 'none'
        else:
            self.ad_type = configs.tta.tsa.ad_type #args['test.tsa_ad_type']
            self.ad_form = configs.tta.tsa.ad_form
        if self.ad_type == 'residual':
            if self.ad_form == 'matrix' or planes != in_planes:
                self.alpha = nn.Parameter(torch.ones(planes, in_planes, 1, 1))
            else:
                self.alpha = nn.Parameter(torch.ones(1, planes, 1, 1))
        elif self.ad_type == 'serial':
            if self.ad_form == 'matrix':
                self.alpha = nn.Parameter(torch.ones(planes, planes, 1, 1))
            else:
                self.alpha = nn.Parameter(torch.ones(1, planes, 1, 1))
            self.alpha_bias = nn.Parameter(torch.ones(1, planes, 1, 1))
            self.alpha_bias.requires_grad = True
        if self.ad_type != 'none':
            self.alpha.requires_grad = True

    def forward(self, x):
        y = self.conv(x)
        if self.ad_type == 'residual':
            raise Exception
            if self.alpha.size(0) > 1:
                # residual adaptation in matrix form
                y = y + F.conv2d(x, self.alpha, stride=self.conv.stride)
            else:
                # residual adaptation in channel-wise (vector)
                y = y + x * self.alpha
        elif self.ad_type == 'serial':
            raise Exception
            if self.alpha.size(0) > 1:
                # serial adaptation in matrix form
                y = F.conv2d(y, self.alpha) + self.alpha_bias
            else:
                # serial adaptation in channel-wise (vector)
                y = y * self.alpha + self.alpha_bias
        return y


class PA(nn.Module):
    """
    pre-classifier alignment (PA) mapping from 'Universal Representation Learning from Multiple Domains for Few-shot Classification'
    (https://arxiv.org/pdf/2103.13841.pdf)
    """
    def __init__(self, feat_dim):
        super(PA, self).__init__()
        # define pre-classifier alignment mapping
        self.weight = nn.Parameter(torch.ones(feat_dim, feat_dim, 1, 1))
        self.weight.requires_grad = True

    def forward(self, x):
        if len(list(x.size())) == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.conv2d(x, self.weight.to(x.device)).flatten(1)
        return x


class TSAResNet(nn.Module):
    """ Attaching task-specific adapters (alpha) and/or PA (beta) to the ResNet backbone """
    def __init__(self, orig_resnet, configs):
        super(TSAResNet, self).__init__()
        self.tsa_init = configs.tta.tsa.init #args['test.tsa_init']
        self.ad_type = configs.tta.tsa.ad_type # args['test.tsa_ad_type']
        # freeze the pretrained backbone
        for k, v in orig_resnet.named_parameters():
                v.requires_grad=False

        # attaching task-specific adapters (alpha) to each convolutional layers
        # note that we only attach adapters to residual blocks in the ResNet
        for block in orig_resnet.layer1:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = TSAConvBlock(m,configs)
                    setattr(block, name, new_conv)

        for block in orig_resnet.layer2:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = TSAConvBlock(m,configs)
                    setattr(block, name, new_conv)

        for block in orig_resnet.layer3:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = TSAConvBlock(m,configs)
                    setattr(block, name, new_conv)

        for block in orig_resnet.layer4:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = TSAConvBlock(m,configs)
                    setattr(block, name, new_conv)

        self.backbone = orig_resnet

        # attach pre-classifier alignment mapping (beta)
        feat_dim = orig_resnet.layer4[-1].bn2.num_features
        beta = PA(feat_dim)
        setattr(self, 'beta', beta)

    def forward(self, x):
        return self.backbone.forward(x=x)

    def embed(self, x):
        return self.backbone.embed(x)

    def get_state_dict(self):
        """Outputs all the state elements"""
        return self.backbone.state_dict()

    def get_parameters(self):
        """Outputs all the parameters"""
        return [v for k, v in self.backbone.named_parameters()]

    def reset(self):

        # initialize task-specific adapters (alpha)
        for k, v in self.backbone.named_parameters():
            if 'alpha' in k:
                # initialize each adapter as an identity matrix
                if self.tsa_init == 'eye':
                    if v.size(0) > 1:
                        v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)
                    else:
                        v.data = torch.ones(v.size()).to(v.device)
                    # for residual adapter, each adapter is initialized as identity matrix scaled by 0.0001
                    if  self.ad_type == 'residual':
                        v.data = v.data * 0.0001
                    if 'bias' in k:
                        v.data = v.data * 0
                elif self.tsa_init == 'random':
                    # randomly initialization
                    v.data = torch.rand(v.data.size()).data.normal_(0, 0.001).to(v.device)
        # initialize pre-classifier alignment mapping (beta)
        v = self.beta.weight
        self.beta.weight.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)


def resnet18(pretrained=True, pretrained_model_path=None,**kwargs):
    print(f'{f"Creating RESNET18 backbone" :=^50}')
    training = kwargs.get('training',None)
    """resent19 backone without BN parameters"""
    nl = partial(nn.BatchNorm2d, track_running_stats=not training)
    model = ResNet(BasicBlock, [2, 2, 2, 2], nl)
    if pretrained:
        print('using pretrained resnet-18')
        # load_model
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict(checkpoint['state_dict'],strict=not training)
        if not training:
            # disable training
            for p in model.parameters():
                p.requires_grad = False
            # use pre-trained batch statistics
            print('bn is set in EVALUATION mode.')
            model.eval()
        else:
            print('Enable fine-tuning.')
            print('bn is set in TRAINING mode.')
            model.train()
    else:
        print('build resnet-18 from scratch')
        # meta batch-norm
        model.train()
    print(f'{f"" :=^50}\n')
    return model

def film_resnet18(pretrained=False, pretrained_model_path=None, **kwargs):
    """
        Constructs a FiLM adapted ResNet-18 model.
    """
    print(f'{f"Creating FILM + RESNET18 backbone" :=^50}')
    bn = partial(nn.BatchNorm2d, track_running_stats=not kwargs.get('training'))
    # bn = nn.BatchNorm2d  # always use batch norm #
    model = FilmResNetOriginal(BasicBlockFilmOriginal, [2, 2, 2, 2], bn)
    # if pretrained:
    #    ckpt_dict = torch.load(pretrained_model_path)
    #    model.load_state_dict(ckpt_dict['state_dict'], strict=False)
    # return model
    if pretrained :
        assert pretrained_model_path, f"unspecified pretrained model path : {pretrained_model_path}"
        ckpt_dict = torch.load(pretrained_model_path)['state_dict']
        shared_state = {k: v for k, v in ckpt_dict.items() if 'cls' not in k}
        missing, unexpected  = model.load_state_dict(shared_state, strict=False)
        print('Loaded shared weights from {}'.format(pretrained_model_path))
        print('the following pretrained parameters are unused : \n', [x for x in unexpected])
        if kwargs.get('training'):
            print('Enable fine-tuning.')
            print('bn is set in TRAINING mode.')
            model.train()
            for p in model.parameters():
                p.requires_grad = True
        else:
            print('bn is set in EVALUATION mode.')
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
    else:
        print('build resnet-18 with film from scratch')
        # meta batch-norm
        model.train()
    print(f'{f"" :=^50}\n')
    return model

def tsa_resnet18(pretrained=False, pretrained_model_path=None, **kwargs):
    print(f'{f"Creating TSA + RESNET18 backbone" :=^50}')
    bn = nn.BatchNorm2d
    model = ResNet(BasicBlock, [2, 2, 2, 2], bn)

    if pretrained:
        assert pretrained_model_path, f"unspecified pretrained model path : {pretrained_model_path}"
        ckpt_dict = torch.load(pretrained_model_path)['state_dict']
        shared_state = {k: v for k, v in ckpt_dict.items() if 'cls' not in k}
        missing, unexpected = model.load_state_dict(shared_state, strict=False)
        print('Loaded shared weights from {}'.format(pretrained_model_path))
        print('the following pretrained parameters are unused : \n', [x for x in unexpected])
        if kwargs.get('training'):
            print('Enable fine-tuning.')
            print('bn is set in TRAINING mode.')
            model.train()
            for p in model.parameters():
                p.requires_grad = True
        else:
            print('bn is set in EVALUATION mode.')
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
    if kwargs.get('tsa_wrapper'):
        model = TSAResNet(orig_resnet=model, configs=kwargs['configs'])
        model.eval()
        print('attached TSA to backbone.')
    else:
        print('No TSA, euivalent to using standard RESNET18.')

    print(f'{f"" :=^50}\n')
    return model


