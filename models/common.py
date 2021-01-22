import cv2
import numpy as np
import torch
import torch.nn as nn

class ReLU1(torch.nn.modules.activation.Hardtanh):
    def __init__(self, inplace: bool = False):
        super(ReLU1, self).__init__(0., 1., inplace)
    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class Blur(nn.Module):
    def __init__(self, weight=None, ksize=17, sigma=3):
        super(Blur, self).__init__()
        layers = []
        layers += [nn.ReflectionPad2d(ksize//2)]
        layers += [nn.Conv2d(3, 3, ksize, stride=1, padding=0, bias=None, groups=3)]
        self.blur = nn.Sequential(*layers)
        self.weight_init(weight, ksize, sigma)
            
    def forward(self, img):
        return self.blur(img)
    
    def weight_init(self, weight=None, ksize=17, sigma=3):
        if weight is None:
            kernel1d = cv2.getGaussianKernel(ksize=ksize, sigma=sigma)
            kernel2d = np.outer(kernel1d, kernel1d.transpose())
            weight = torch.from_numpy(kernel2d)
        for name, param in self.named_parameters():
            param.data.copy_(weight)
            
class Edge(torch.nn.Module):
    def __init__(self, num_fea=3, weight=None):
        super(Edge, self).__init__()
        layers = []
        layers += [torch.nn.ReflectionPad2d(1)]
        layers += [torch.nn.Conv2d(num_fea, num_fea, 3, stride=1, padding=0, bias=None, groups=num_fea)]
        self.blur = torch.nn.Sequential(*layers)
        self.weight_init(weight)
            
    def forward(self, img):
        return self.blur(img)
    
    def weight_init(self, weight):
        if weight is None:
            kernel = torch.tensor([[[-1.000, -1.000, -1.000],
                                    [-1.000, 8.0000, -1.000],
                                    [-1.000, -1.000, -1.000]]])
        for name, param in self.named_parameters():
            param.data.copy_(kernel)
            param.requires_grad = False
            
class Prewitt(nn.Module):
    def __init__(self, mode='x', pad=1):
        super(Prewitt, self).__init__()
        num_fea = 3
        layers = [nn.ReflectionPad2d(pad),
                  nn.Conv2d(num_fea, num_fea, 3, stride=1, padding=0, bias=None, groups=num_fea)]
        self.blur = nn.Sequential(*layers)
        self.weight_init(mode)
            
    def forward(self, img):
        return self.blur(img)
    
    def weight_init(self, mode):
        if mode == 'x':
            kernel = torch.tensor([[[1, 0, -1],
                                    [1, 0, -1],
                                    [1, 0, -1]]]) / 3
        elif mode == 'y':
            kernel = torch.tensor([[[ 1,  1,  1],
                                    [ 0,  0,  0],
                                    [-1, -1, -1]]]) / 3
        for name, param in self.named_parameters():
            param.data.copy_(kernel)
            param.requires_grad = False
            
            
class GMSD(nn.Module):
    def __init__(self, rgb_scale=1, pad=1):
        super(GMSD, self).__init__()
        self.rgb_scale = rgb_scale
        self.prewitt_x = Prewitt('x', pad=pad)
        self.prewitt_y = Prewitt('y', pad=pad)
    def forward(self, img):
        px = self.prewitt_x(img * (255/self.rgb_scale))
        py = self.prewitt_y(img * (255/self.rgb_scale))
        return torch.sqrt(torch.square(px) + torch.square(py)) * (self.rgb_scale/255)
            
    
class GMSD_quality(nn.Module):
    def __init__(self, rgb_scale=1):
        super(GMSD_quality, self).__init__()
        self.GMSD = GMSD()
        self.T = 170.
        self.rgb_scale = rgb_scale
    def forward(self, img1, img2):
        img1 = img1 * (255/self.rgb_scale)
        img2 = img2 * (255/self.rgb_scale)
        gm1 = self.GMSD(img1)
        gm2 = self.GMSD(img2)
        return 1. - (2. * gm1 * gm2 + self.T) / (torch.square(gm1) + torch.square(gm2) + self.T)
        
        
class ReceptiveFieldBlock(nn.Module):
    """ from RFB-ESRGAN """
    def __init__(self, in_channels, out_channels, scale_ratio=0.2, non_linearity=True):
        super(ReceptiveFieldBlock, self).__init__()
        # channels = in_channels // 4
        channels = in_channels
        # shortcut layer
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=3, dilation=3, bias=False)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=3, dilation=3, bias=False)
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 3, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 3, channels, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=5, dilation=5, bias=False)
        )

        self.conv1x1 = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True) if non_linearity else None

        self.scale_ratio = scale_ratio

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        shortcut = self.shortcut(x)

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        out = torch.cat((branch1, branch2, branch3, branch4), 1)
        out = self.conv1x1(out)

        out = out.mul(self.scale_ratio) + shortcut
        if self.lrelu is not None:
            out = self.lrelu(out)

        return out
        
        
class SecondOrder(torch.nn.Module):
    def __init__(self, in_channels=64, out_channels=64, ksize=3, mode='h'):
        super(SecondOrder, self).__init__()
        layers = [torch.nn.ReflectionPad2d(ksize//2),
                  torch.nn.Conv2d(in_channels, out_channels, ksize, stride=1, padding=0, bias=None, groups=out_channels)]
        self.conv = torch.nn.Sequential(*layers)
        self.weight_init(ksize, mode)
            
    def forward(self, img):
        return self.conv(img)
    
    def weight_init(self, ksize, mode):
        kernel = torch.zeros((1, ksize, ksize))
        if mode=='h':
            kernel[0][ksize//2][0] = 1
            kernel[0][ksize//2][ksize//2] = -2
            kernel[0][ksize//2][-1] = 1
        elif mode=='v':
            kernel[0][0][ksize//2] = 1
            kernel[0][ksize//2][ksize//2] = -2
            kernel[0][-1][ksize//2] = 1
        else:
            kernel[0][0][0] = 1
            kernel[0][0][-1] = -1
            kernel[0][-1][0] = -1
            kernel[0][-1][-1] = 1
        for name, param in self.named_parameters():
            param.data.copy_(kernel)
            param.requires_grad = False
            
class MSHF(torch.nn.Module):
    def __init__(self, in_channels=64, out_channels=64, rgb_scale=1):
        super(MSHF, self).__init__()
        
        scale_3 = [SecondOrder(in_channels=in_channels, out_channels=out_channels, ksize=3, mode=mode) for mode in ['h', 'v', 'hv']]
        scale_5 = [SecondOrder(in_channels=in_channels, out_channels=out_channels, ksize=5, mode=mode) for mode in ['h', 'v', 'hv']]
        scale_7 = [SecondOrder(in_channels=in_channels, out_channels=out_channels, ksize=7, mode=mode) for mode in ['h', 'v', 'hv']]
        
        self.scale_3 = nn.ModuleList(scale_3)
        self.scale_5 = nn.ModuleList(scale_5)
        self.scale_7 = nn.ModuleList(scale_7)
        self.rgb_scale = rgb_scale
        self.act = ReLU1()
        self.eps = 1e-7
        
    def forward(self, x):
        x = x * (255 / self.rgb_scale)
    
        hh = self.scale_3[0](x)
        vv = self.scale_3[1](x)
        hv = self.scale_3[2](x)        
        fea = torch.square(hh - vv) + 4 * torch.square(hv)
        fea = torch.sqrt(fea + self.eps)        
        fea_3 = (hh + vv + fea) * 0.5
        fea_3 = torch.mean(fea_3, dim=1, keepdim=True)
        
        hh = self.scale_5[0](x)
        vv = self.scale_5[1](x)
        hv = self.scale_5[2](x)        
        fea = torch.square(hh - vv) + 4 * torch.square(hv)
        fea = torch.sqrt(fea + self.eps)        
        fea_5 = (hh + vv + fea) * 0.5
        fea_5 = torch.mean(fea_5, dim=1, keepdim=True)
        
        hh = self.scale_7[0](x)
        vv = self.scale_7[1](x)
        hv = self.scale_7[2](x)        
        fea = torch.square(hh - vv) + 4 * torch.square(hv)
        fea = torch.sqrt(fea + self.eps)        
        fea_7 = (hh + vv + fea) * 0.5
        fea_7 = torch.mean(fea_7, dim=1, keepdim=True)
        
        fea = torch.cat((fea_3, fea_5, fea_7), dim=1)
        
        fea = fea * (self.rgb_scale / 255)
        return self.act(fea)
        
        
from torchvision.models import vgg16

class VGG_Extractor(nn.Module):
    def __init__(self):
        super(VGG_Extractor, self).__init__()
        self.vgg = vgg16(pretrained=True)
        
    def forward(self, x):
        out_x = list()
        for vgg_layer in list(self.vgg.features.children()):
            if type(vgg_layer) == torch.nn.modules.pooling.MaxPool2d:
                out_x.append(x)
            x = vgg_layer.forward(x)
        return out_x
