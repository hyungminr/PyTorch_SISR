import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import sigmoid
import torch.nn.functional as F
import math


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class kerAdp_block(nn.Module):
    def __init__(self, nf=32, para=1):
        super(kerAdp_block, self).__init__()
        self.down = nn.Sequential(nn.Conv2d(nf, nf*2, kernel_size=3,stride=2,padding=1),
                                  nn.LeakyReLU(0.2),
                                  nn.Conv2d(nf*2, nf*4, kernel_size=3,stride=2,padding=1),
                                  AdaptiveInstanceNorm2d(nf*4),
                                  nn.LeakyReLU(0.2)
                                 )
        self.up = nn.Sequential(nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4,stride=2,padding=1),
                                  nn.LeakyReLU(0.2),
                                nn.ConvTranspose2d(nf*2, nf, kernel_size=4,stride=2,padding=1),
                                  nn.LeakyReLU(0.2)
                                 )
    
        self.conv1_mul = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.conv1_plus = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        
        self.mlp = MLP(289, self.get_num_adain_params(self.down), 256, 3, norm='none')
        
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, feature_maps, para_maps):
        adain_params = self.mlp(para_maps)
        self.assign_adain_params(adain_params, self.down)
        cat_input = self.down(feature_maps)
    
        cat_input = self.up(cat_input)
    
        cat_mul = self.sig(self.conv1_mul(cat_input))
        cat_plus = self.conv1_plus(cat_input)
    
        cat_input = cat_mul * feature_maps + cat_plus
        
        return cat_input
        
    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params
    
    

class Deblurring_Block(nn.Module):
    def __init__(self, nf=64):
        super(Deblurring_Block, self).__init__()
        self.kab1 = kerAdp_block(nf)
        self.kab2 = kerAdp_block(nf)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, feature_maps, para_maps):
        fea1 = F.relu(self.kab1(feature_maps, para_maps))
        fea2 = F.relu(self.kab2(self.conv1(fea1), para_maps))
        fea3 = self.conv2(fea2)
        return torch.add(feature_maps, fea3)
    
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range=1.0, rgb_mean=(0.4488,0.4371,0.4040), rgb_std=(1.0,1.0,1.0), mode='sub'):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        sign = -1 if mode == 'sub' else 1
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.weight.requires_grad = False
        self.bias.requires_grad = False

class BIKA(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, input_para=10, scale=2):
        super(BIKA, self).__init__()
        self.para = input_para
        self.num_blocks = nb
        self.in_nc = in_nc
        self.scale = scale
        
        self.conv1 = nn.Conv2d(in_nc, nf, 3, stride=1, padding=1)
        self.relu_conv1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(nf, nf, 3, stride=1, padding=1)
        self.relu_conv2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(nf, nf, 3, stride=1, padding=1)
        

        for i in range(nb):
            self.add_module('Dbr-block' + str(i + 1), Deblurring_Block(nf=nf))

        self.sft = kerAdp_block(nf=nf)
        self.conv_mid = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_mid_lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv_output = nn.Conv2d(in_channels=nf, out_channels=out_nc, kernel_size=9, stride=1, padding=4, bias=True)
        
        num_feats = nf
        
        kernel=3
        padding=1
        bias=True
        
        layers = []
        if (scale & (scale - 1)) == 0: # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                layers += [nn.Conv2d(in_channels=num_feats, out_channels=num_feats*4, kernel_size=kernel, padding=padding, bias=bias)]
                layers += [nn.PixelShuffle(2)]
        layers += [nn.Conv2d(in_channels=num_feats, out_channels=3, kernel_size=kernel, padding=padding, bias=bias)]
        self.tail = nn.Sequential(*layers)
        
        self.sub_mean = MeanShift(mode='sub')
        self.add_mean = MeanShift(mode='add')
        
        self.up = nn.UpsamplingNearest2d(scale_factor=self.scale)
        
        
    def forward(self, img):
        B, C, H, W = img.size()
        ker_code_exp = torch.zeros((B, 1, 17, 17), device=img.device)
        img_up = self.up(img)
        
        x = self.sub_mean(img)
        
        fea_bef = self.conv3(self.relu_conv2(self.conv2(self.relu_conv1(self.conv1(x)))))
        
        fea_in = fea_bef
        for i in range(self.num_blocks):
            fea_in = self.__getattr__('Dbr-block' + str(i + 1))(fea_in, ker_code_exp)
            fea_in = torch.add(fea_in, fea_bef)
            
        fea_add = torch.add(fea_in, fea_bef)
        out = self.conv_mid_lrelu(self.conv_mid(self.sft(fea_add, ker_code_exp)))
        
        
        out_deb = self.conv_output(out)        
        out_deb = self.add_mean(out_deb)        
        deblurred = out_deb + img
        
        out_sup = self.tail(out)
        out_sup = self.add_mean(out_sup)
        superred = out_sup + img_up
        
        return superred, deblurred
    
class mapping_net(nn.Module):
    def __init__(self, in_nc=1, nf=1, code_len=1, use_bias=False):
        super(mapping_net, self).__init__()
        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(in_nc, 16, kernel_size=3, stride=1, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 289, kernel_size=3, stride=1, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(7)
        ])
    def forward(self, input):
        conv = self.ConvNet(input)
        return conv 

    

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='lrelu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
    
class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

    
    
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'
