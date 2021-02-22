
import torch
import torch.nn as nn
import torch.nn.init as init

class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=3, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
                
                
import torch
import torch.nn as nn
import torch.nn.functional as F

class MemNet(nn.Module):
    def __init__(self, in_channels=3, channels=64, num_memblock=6, num_resblock=6):
        super(MemNet, self).__init__()
        self.feature_extractor = BNReLUConv(in_channels, channels)
        self.reconstructor = BNReLUConv(channels, in_channels)
        self.dense_memory = nn.ModuleList(
            [MemoryBlock(channels, num_resblock, i+1) for i in range(num_memblock)]
        )

    def forward(self, x):
        # x = x.contiguous()
        residual = x
        out = self.feature_extractor(x)
        ys = [out]
        for memory_block in self.dense_memory:
            out = memory_block(out, ys)
        out = self.reconstructor(out)
        out = out + residual
        
        return out


class MemoryBlock(nn.Module):
    """Note: num_memblock denotes the number of MemoryBlock currently"""
    def __init__(self, channels, num_resblock, num_memblock):
        super(MemoryBlock, self).__init__()
        self.recursive_unit = nn.ModuleList(
            [ResidualBlock(channels) for i in range(num_resblock)]
        )
        self.gate_unit = BNReLUConv((num_resblock+num_memblock) * channels, channels, 1, 1, 0)

    def forward(self, x, ys):
        """ys is a list which contains long-term memory coming from previous memory block
        xs denotes the short-term memory coming from recursive unit
        """
        xs = []
        residual = x
        for layer in self.recursive_unit:
            x = layer(x)
            xs.append(x)
        
        gate_out = self.gate_unit(torch.cat(xs+ys, 1))
        ys.append(gate_out)
        return gate_out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    x - Relu - Conv - Relu - Conv - x
    """

    def __init__(self, channels, k=3, s=1, p=1):
        super(ResidualBlock, self).__init__()
        self.relu_conv1 = BNReLUConv(channels, channels, k, s, p)
        self.relu_conv2 = BNReLUConv(channels, channels, k, s, p)
        
    def forward(self, x):
        residual = x
        out = self.relu_conv1(x)
        out = self.relu_conv2(out)
        out = out + residual
        return out


class BNReLUConv(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=True):
        super(BNReLUConv, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))
        self.add_module('conv', nn.Conv2d(in_channels, channels, k, s, p, bias=False))
        
        
        
        
        
        

class _DCR_block(nn.Module):
    def __init__(self, channel_in):
        super(_DCR_block, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=channel_in, out_channels=int(channel_in/2.), kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.PReLU()
        self.conv_2 = nn.Conv2d(in_channels=int(channel_in*3/2.), out_channels=int(channel_in/2.), kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.PReLU()
        self.conv_3 = nn.Conv2d(in_channels=channel_in*2, out_channels=channel_in, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.PReLU()

    def forward(self, x):
        residual = x

        out = self.relu1(self.conv_1(x))

        conc = torch.cat([x, out], 1)

        out = self.relu2(self.conv_2(conc))

        conc = torch.cat([conc, out], 1)

        out = self.relu3(self.conv_3(conc))

        out = torch.add(out, residual)

        return out


class _down(nn.Module):
    def __init__(self, channel_in):
        super(_down, self).__init__()

        self.relu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=2*channel_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.maxpool(x)

        out = self.relu(self.conv(out))

        return out


class _up(nn.Module):
    def __init__(self, channel_in):
        super(_up, self).__init__()

        self.relu = nn.PReLU()
        self.subpixel = nn.PixelShuffle(2)
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.relu(self.conv(x))

        out = self.subpixel(out)

        return out

class DHDN(nn.Module):
    def __init__(self):
        super(DHDN, self).__init__()

        self.conv_i = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.PReLU()
        self.DCR_block11 = self.make_layer(_DCR_block, 128)
        self.DCR_block12 = self.make_layer(_DCR_block, 128)
        self.down1 = self.make_layer(_down, 128)
        self.DCR_block21 = self.make_layer(_DCR_block, 256)
        self.DCR_block22 = self.make_layer(_DCR_block, 256)
        self.down2 = self.make_layer(_down, 256)
        self.DCR_block31 = self.make_layer(_DCR_block, 512)
        self.DCR_block32 = self.make_layer(_DCR_block, 512)
        self.down3 = self.make_layer(_down, 512)
        self.DCR_block41 = self.make_layer(_DCR_block, 1024)
        self.DCR_block42 = self.make_layer(_DCR_block, 1024)
        self.up3 = self.make_layer(_up, 2048)
        self.DCR_block33 = self.make_layer(_DCR_block, 1024)
        self.DCR_block34 = self.make_layer(_DCR_block, 1024)
        self.up2 = self.make_layer(_up, 1024)
        self.DCR_block23 = self.make_layer(_DCR_block, 512)
        self.DCR_block24 = self.make_layer(_DCR_block, 512)
        self.up1 = self.make_layer(_up, 512)
        self.DCR_block13 = self.make_layer(_DCR_block, 256)
        self.DCR_block14 = self.make_layer(_DCR_block, 256)
        self.conv_f = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.PReLU()

    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x

        out = self.relu1(self.conv_i(x))

        out = self.DCR_block11(out)

        conc1 = self.DCR_block12(out)

        out = self.down1(conc1)

        out = self.DCR_block21(out)

        conc2 = self.DCR_block22(out)

        out = self.down2(conc2)

        out = self.DCR_block31(out)

        conc3 = self.DCR_block32(out)

        conc4 = self.down3(conc3)

        out = self.DCR_block41(conc4)

        out = self.DCR_block42(out)

        out = torch.cat([conc4, out], 1)

        out = self.up3(out)

        out = torch.cat([conc3, out], 1)

        out = self.DCR_block33(out)

        out = self.DCR_block34(out)

        out = self.up2(out)

        out = torch.cat([conc2, out], 1)

        out = self.DCR_block23(out)

        out = self.DCR_block24(out)

        out = self.up1(out)

        out = torch.cat([conc1, out], 1)

        out = self.DCR_block13(out)

        out = self.DCR_block14(out)

        out = self.relu2(self.conv_f(out))

        out = torch.add(residual, out)

        return out
        
        
def FFDNet_downsample(x):
    """
    :param x: (C, H, W)
    :param noise_sigma: (C, H/2, W/2)
    :return: (4, C, H/2, W/2)
    """
    # x = x[:, :, :x.shape[2] // 2 * 2, :x.shape[3] // 2 * 2]
    N, C, W, H = x.size()
    idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]

    Cout = 4 * C
    Wout = W // 2
    Hout = H // 2

    if 'cuda' in x.type():
        down_features = torch.cuda.FloatTensor(N, Cout, Wout, Hout).fill_(0)
    else:
        down_features = torch.FloatTensor(N, Cout, Wout, Hout).fill_(0)
    
    for idx in range(4):
        down_features[:, idx:Cout:4, :, :] = x[:, :, idxL[idx][0]::2, idxL[idx][1]::2]

    return down_features

def FFDNet_upsample(x):
    """
    :param x: (n, C, W, H)
    :return: (n, C/4, W*2, H*2)
    """
    N, Cin, Win, Hin = x.size()
    idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]
    
    Cout = Cin // 4
    Wout = Win * 2
    Hout = Hin * 2

    up_feature = torch.zeros((N, Cout, Wout, Hout)).type(x.type())
    for idx in range(4):
        up_feature[:, :, idxL[idx][0]::2, idxL[idx][1]::2] = x[:, idx:Cin:4, :, :]

    return up_feature

import numpy as np
from torch.autograd import Variable
class FFDNet(nn.Module):

    def __init__(self, is_gray=False):
        super(FFDNet, self).__init__()

        if is_gray:
            self.num_conv_layers = 15 # all layers number
            self.downsampled_channels = 5 # Conv_Relu in
            self.num_feature_maps = 64 # Conv_Bn_Relu in
            self.output_features = 4 # Conv out
        else:
            self.num_conv_layers = 12
            self.downsampled_channels = 15
            self.num_feature_maps = 96
            self.output_features = 12
            
        self.kernel_size = 3
        self.padding = 1
        
        layers = []
        # Conv + Relu
        layers.append(nn.Conv2d(in_channels=self.downsampled_channels, out_channels=self.num_feature_maps, \
                                kernel_size=self.kernel_size, padding=self.padding, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Conv + BN + Relu
        for _ in range(self.num_conv_layers - 2):
            layers.append(nn.Conv2d(in_channels=self.num_feature_maps, out_channels=self.num_feature_maps, \
                                    kernel_size=self.kernel_size, padding=self.padding, bias=False))
            layers.append(nn.BatchNorm2d(self.num_feature_maps))
            layers.append(nn.ReLU(inplace=True))
        
        # Conv
        layers.append(nn.Conv2d(in_channels=self.num_feature_maps, out_channels=self.output_features, \
                                kernel_size=self.kernel_size, padding=self.padding, bias=False))

        self.intermediate_dncnn = nn.Sequential(*layers)

    def forward(self, x):
        noise_sigma = torch.FloatTensor(np.array([30/255 for idx in range(x.shape[0])]))
        noise_map = noise_sigma.view(x.shape[0], 1, 1, 1).repeat(1, x.shape[1], x.shape[2] // 2, x.shape[3] // 2).to(x.device)

        x_up = FFDNet_downsample(x.data) # 4 * C * H/2 * W/2
        x_cat = torch.cat((noise_map.data, x_up), 1) # 4 * (C + 1) * H/2 * W/2
        x_cat = Variable(x_cat)

        h_dncnn = self.intermediate_dncnn(x_cat)
        y_pred = FFDNet_upsample(h_dncnn)
        return y_pred
