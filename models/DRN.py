import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return DRN(args)

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
    def __init__(self, num_fea=64, reduction=16, kernel=3, padding=1, bias=True):
        super().__init__()
        
        layers = []
        layers += [nn.Conv2d(in_channels=num_fea, out_channels=num_fea, kernel_size=kernel, padding=padding, bias=bias)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(in_channels=num_fea, out_channels=num_fea, kernel_size=kernel, padding=padding, bias=bias)]
        self.conv_in = nn.Sequential(*layers)
        
        layers = []
        layers += [nn.AdaptiveAvgPool2d(1)]
        layers += [nn.Conv2d(in_channels=num_fea, out_channels=num_fea//reduction, kernel_size=kernel, padding=padding, bias=bias)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(in_channels=num_fea//reduction, out_channels=num_fea, kernel_size=kernel, padding=padding, bias=bias)]
        layers += [nn.Sigmoid()]
        self.channel_att = nn.Sequential(*layers)
        
    def forward(self, x):
        x_fea = self.conv_in(x)
        x_cha = self.channel_att(x_fea)
        x_att = x_fea * x_cha
        return x + x_att
    
class Down(nn.Module):
    def __init__(self, n_feats=16, negval=0.2):
        super().__init__()
        layers = [nn.Conv2d(in_channels=3, out_channels=n_feats, kernel_size=3, padding=1, stride=2, bias=False)]
        layers += [nn.LeakyReLU(negative_slope=negval, inplace=True)]
<<<<<<< HEAD
=======
<<<<<<< HEAD
        layers += [nn.Conv2d(in_channels=n_feats, out_channels=3 kernel_size=3, padding=1, bias=False)]
        self.down = nn.Sequential(*layers)
    def forward(self, img):
        imgx4 = self.down(img)
=======
>>>>>>> e6994fa56381720ca2a9602b8f226e8edfaad487
        layers += [nn.Conv2d(in_channels=n_feats, out_channels=3, kernel_size=3, padding=1, bias=False)]
        self.down = nn.Sequential(*layers)
    def forward(self, img):
        return self.down(img)
<<<<<<< HEAD
=======
>>>>>>> b76c98a3918208af5db150361889dd3791c727f8
>>>>>>> e6994fa56381720ca2a9602b8f226e8edfaad487
    
class DRN(nn.Module):
    """  """
    def __init__(self, scale=[2, 4], n_blocks=30, n_feats=16, negval=0.2):
        super().__init__()
        self.scale = [2, 4]
        self.n_blocks = 30
        self.n_feats = 16
        self.negval = 0.2
        
        kernel_size = 3
        """ """
        self.up_bicubic = nn.Upsample(scale_factor=max(self.scale), mode='bicubic', align_corners=False)
        """ """
        self.sub_mean = MeanShift(mode='sub')
        """ """
        layers = [nn.Conv2d(in_channels=3, out_channels=self.n_feats, kernel_size=kernel_size, padding=kernel_size//2, bias=True)]
        self.head = nn.Sequential(*layers)
        """ """
        # phase 0
        layers = [nn.Conv2d(in_channels=self.n_feats, out_channels=self.n_feats, kernel_size=kernel_size, padding=kernel_size//2, stride=2, bias=False)]
        layers += [nn.LeakyReLU(negative_slope=self.negval, inplace=True)]
        layers += [nn.Conv2d(in_channels=self.n_feats, out_channels=self.n_feats*2, kernel_size=kernel_size, padding=kernel_size//2, bias=False)]
        self.down_0 = nn.Sequential(*layers)
        # phase 1
        layers = [nn.Conv2d(in_channels=self.n_feats*2, out_channels=self.n_feats*2, kernel_size=kernel_size, padding=kernel_size//2, stride=2, bias=False)]
        layers += [nn.LeakyReLU(negative_slope=self.negval, inplace=True)]
        layers += [nn.Conv2d(in_channels=self.n_feats*2, out_channels=self.n_feats*4, kernel_size=kernel_size, padding=kernel_size//2, bias=False)]
        self.down_1 = nn.Sequential(*layers)
        """ """   
        # phase 0
        layers = [RCAB(num_fea=self.n_feats*4, kernel=kernel_size, padding=kernel_size//2, bias=True) for _ in range(self.n_blocks)]
        layers += [nn.Conv2d(in_channels=self.n_feats*4, out_channels=self.n_feats*16, kernel_size=kernel_size, padding=kernel_size//2, bias=True)]
        layers += [nn.PixelShuffle(2)]
        layers += [nn.Conv2d(in_channels=self.n_feats*4, out_channels=self.n_feats*2, kernel_size=1, bias=True)]
        self.up_block_0 = nn.Sequential(*layers)
        # phase 1
        layers = [RCAB(num_fea=self.n_feats*4, kernel=kernel_size, padding=kernel_size//2, bias=True) for _ in range(self.n_blocks)]
        layers += [nn.Conv2d(in_channels=self.n_feats*4, out_channels=self.n_feats*16, kernel_size=kernel_size, padding=kernel_size//2, bias=True)]
        layers += [nn.PixelShuffle(2)]
        layers += [nn.Conv2d(in_channels=self.n_feats*4, out_channels=self.n_feats, kernel_size=1, bias=True)]
        self.up_block_1 = nn.Sequential(*layers)
        """ """
        layers = [nn.Conv2d(in_channels=self.n_feats*4, out_channels=3, kernel_size=kernel_size, padding=kernel_size//2, bias=True)]
        self.tail_0 = nn.Sequential(*layers)
        layers = [nn.Conv2d(in_channels=self.n_feats*4, out_channels=3, kernel_size=kernel_size, padding=kernel_size//2, bias=True)]
        self.tail_1 = nn.Sequential(*layers)
        layers = [nn.Conv2d(in_channels=self.n_feats*2, out_channels=3, kernel_size=kernel_size, padding=kernel_size//2, bias=True)]
        self.tail_2 = nn.Sequential(*layers)
        """ """
        self.add_mean = MeanShift(mode='add')
        
    def forward(self, imgx1):
        imgx4 = self.up_bicubic(imgx1)
        
        feax4 = self.sub_mean(imgx4)
        feax4 = self.head(feax4)
        
        feax2 = self.down_0(feax4)
        feax1 = self.down_1(feax2)
        
        resx1 = self.tail_0(feax1)
        resx1 = self.add_mean(resx1)
        
        feax2 = torch.cat((feax2, self.up_block_0(feax1)), dim=1)
        resx2 = self.tail_1(feax2)
        resx2 = self.add_mean(resx2)
        
        feax4 = torch.cat((feax4, self.up_block_1(feax2)), dim=1)
        resx4 = self.tail_2(feax4)
        resx4 = self.add_mean(resx4)
        
        return resx1, resx2, resx4
