import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class RCAB(nn.Module):
    """ Residual Channel Attention Block """
    def __init__(self, kernel=3, padding=1, bias=True, num_feats=64):
        super().__init__()
        
        layers = []
        layers += [nn.Conv2d(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel, padding=padding, bias=bias)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel, padding=padding, bias=bias)]
        self.conv_in = nn.Sequential(*layers)
        
        layers = []
        layers += [nn.AdaptiveAvgPool2d(1)]
        layers += [nn.Conv2d(in_channels=num_feats, out_channels= 4, kernel_size=kernel, padding=padding, bias=bias)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(in_channels= 4, out_channels=num_feats, kernel_size=kernel, padding=padding, bias=bias)]
        layers += [nn.Sigmoid()]
        self.channel_att = nn.Sequential(*layers)
        
    def forward(self, x):
        x_fea = self.conv_in(x)
        x_cha = self.channel_att(x_fea)
        x_att = x_fea * x_cha
        return x + x_att

class RG(nn.Module):
    """ Residual Group """
    def __init__(self, num_RCAB=16, kernel=3, padding=1, bias=True, num_feats=64):
        super().__init__()
        
        layers = []
        layers += [RCAB(num_feats=num_feats) for _ in range(num_RCAB)]
        layers += [nn.Conv2d(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel, padding=padding, bias=bias)]
        self.rcab = nn.Sequential(*layers)
        
    def forward(self, x):
        return x + self.rcab(x)
    


class FusionBlock(nn.Module):
    """ Residual Block """
    def __init__(self, kernel=3, in_channels=6, out_channels=3, num_feats=64, padding=1, bias=True, res_scale=1):
        super().__init__()
        
        layers = []
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=num_feats, kernel_size=kernel, padding=padding, bias=bias)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(in_channels=num_feats, out_channels=out_channels, kernel_size=kernel, padding=padding, bias=bias)]
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.conv(x)
        
        
class RCAN(nn.Module):
    """  """
    def __init__(self, num_RG=10, scale=2, kernel=3, padding=1, bias=True, num_feats=64):
        super().__init__()
        
        self.sub_mean = MeanShift(mode='sub')
        
        layers = [nn.Conv2d(in_channels= 3, out_channels=num_feats, kernel_size=kernel, padding=padding, bias=bias)]
        self.head = nn.Sequential(*layers)
        
        blocks = [RG(num_feats=num_feats) for _ in range(num_RG)]
        blocks += [nn.Conv2d(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel, padding=padding, bias=bias)]
        self.body = nn.ModuleList(blocks)
        
        
        layers = []
        if (scale & (scale - 1)) == 0: # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                layers += [nn.Conv2d(in_channels=num_feats, out_channels=num_feats*4, kernel_size=kernel, padding=padding, bias=bias)]
                layers += [nn.PixelShuffle(2)]
        layers += [nn.Conv2d(in_channels=num_feats, out_channels=3, kernel_size=kernel, padding=padding, bias=bias)]
        self.tail = nn.Sequential(*layers)
        
        self.add_mean = MeanShift(mode='add')
        
    def forward(self, img):    
    
        # meanshift (preprocess)
        x = self.sub_mean(img)
        
        # shallow feature
        x_shallow = self.head(x)
        
        # deep feature
        x_deep = [x_shallow]
        for block in self.body:
            x_deep.append(block(x_deep[-1]))
        
        # shallow + deep
        x_feature = x_shallow + x_deep[-1]
        
        # upscale
        x_up = self.tail(x_feature)
        
        # meanshift (postprocess)
        out = self.add_mean(x_up)
        
        return out, x_deep
        

class RCAN_fusion(nn.Module):
    """  """
    def __init__(self, num_RG=10, scale=2, kernel=3, padding=1, bias=True, num_feats=64):
        super().__init__()
        self.RCAN_1 = RCAN(num_RG, scale, kernel, padding, bias, num_feats)
        self.RCAN_2 = RCAN(num_RG, scale, kernel, padding, bias, num_feats)
        self.RCAN_3 = RCAN(num_RG, scale, kernel, padding, bias, num_feats)
        self.Fusion = FusionBlock()
        self.sub_mean = MeanShift(mode='sub')
        self.add_mean = MeanShift(mode='add')
        
    def forward(self, img, hf, lf):
        img, _ = self.RCAN_1(img)
        hf, _  = self.RCAN_2(hf)
        lf, _  = self.RCAN_3(lf)
        
        out = self.Fusion(torch.cat([self.sub_mean(hf), self.sub_mean(lf)], dim=1))
        
        out = img + self.add_mean(out)
        
        return out, img, hf, lf
