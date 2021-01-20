import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return EDSR(args)
    
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

class ResBlock(nn.Module):
    """ Residual Block """
    def __init__(self, kernel=3, padding=1, bias=True, res_scale=1):
        super().__init__()
        
        layers = []
        layers += [nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel, padding=padding, bias=bias)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel, padding=padding, bias=bias)]
        self.conv = nn.Sequential(*layers)
        self.res_scale = res_scale
        
    def forward(self, x):
        res = self.conv(x).mul(self.res_scale)
        return x + res
        
                
class EDSR(nn.Module):
    """  """
    def __init__(self, args=None, num_RB=16, num_feats=64, scale=2, kernel=3, padding=1, bias=True):
        super().__init__()
                
        self.sub_mean = MeanShift(mode='sub')
        layers = []
        layers += [nn.Conv2d(in_channels= 3, out_channels=num_feats, kernel_size=kernel, padding=padding, bias=bias)]
        self.head = nn.Sequential(*layers)
        
        blocks = [ResBlock() for _ in range(16)]
        blocks += [nn.Conv2d(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel, padding=padding, bias=bias)]
        self.body = nn.ModuleList(blocks)
        
        scale = 4
        layers = []
        for _ in range(int(math.log(scale, 2))):
            layers += [nn.Conv2d(in_channels=num_feats, out_channels=num_feats*4, kernel_size=kernel, padding=padding, bias=bias)]
            layers += [nn.PixelShuffle(2)]
        self.upsampler_x4 = nn.Sequential(*layers)
        
        scale = 2
        layers = []
        for _ in range(int(math.log(scale, 2))):
            layers += [nn.Conv2d(in_channels=num_feats, out_channels=num_feats*4, kernel_size=kernel, padding=padding, bias=bias)]
            layers += [nn.PixelShuffle(2)]
        self.upsampler_x2 = nn.Sequential(*layers)
        
        layers = [nn.Conv2d(in_channels=num_feats, out_channels=3, kernel_size=kernel, padding=padding, bias=bias)]
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
        x_feature = x_deep[0] + x_deep[-1]
        
        # upscale
        x_feature_x4 = self.upsampler_x4(x_feature)
        x_feature_x2 = self.upsampler_x2(x_feature)
        
        x_x4 = self.tail(x_feature_x4)
        x_x2 = self.tail(x_feature_x2)
        x_x1 = self.tail(x_feature)
        
        # meanshift (postprocess)
        out_x4 = self.add_mean(x_x4)
        out_x2 = self.add_mean(x_x2)
        out_x1 = self.add_mean(x_x1)
        x_deep = [out_x2, out_x1] + x_deep
        
        return out_x4, x_deep
