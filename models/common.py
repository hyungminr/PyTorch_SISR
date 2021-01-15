import cv2
import numpy as np
import torch
import torch.nn as nn

class Blur(nn.Module):
    def __init__(self, weight=None):
        super(Blur, self).__init__()
        layers = []
        layers += [nn.ReflectionPad2d(8)]
        layers += [nn.Conv2d(3, 3, 17, stride=1, padding=0, bias=None, groups=3)]
        self.blur = nn.Sequential(*layers)
        self.weight_init(weight)
            
    def forward(self, img):
        return self.blur(img)
    
    def weight_init(self, weight=None, ksize=17, sigma=3):
        print(weight)
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
    def __init__(self, mode='x'):
        super(Prewitt, self).__init__()
        num_fea = 3
        layers = [nn.ReflectionPad2d(1),
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
    def __init__(self):
        super(GMSD, self).__init__()
        self.prewitt_x = Prewitt('x')
        self.prewitt_y = Prewitt('y')
    def forward(self, img):
        rgb_scale = 255 if img.max() > 2 else 1
        px = self.prewitt_x(img * (255/rgb_scale))
        py = self.prewitt_y(img * (255/rgb_scale))
        return torch.sqrt(torch.square(px) + torch.square(py)) * (rgb_scale/255)
    
class GMSD_quality(nn.Module):
    def __init__(self, rgb_scale=1):
        super(GMSD_quality, self).__init__()
        self.GMSD = GMSD()
        self.T = 170.
        self.rgb_scale = rgb_scale
    def forward(self, img1, img2):
        if self.rgb_scale == 1:
            img1 = img1 * 255.
            img2 = img2 * 255.
        gm1 = self.GMSD(img1)
        gm2 = self.GMSD(img2)
        return 1. - (2. * gm1 * gm2 + self.T) / (torch.square(gm1) + torch.square(gm2) + self.T)
                
class ReceptiveFieldBlock(nn.Module):
    """ from RFB-ESRGAN """
    def __init__(self, in_channels, out_channels, scale_ratio=0.2, non_linearity=True):
        super(ReceptiveFieldBlock, self).__init__()
        channels = in_channels // 4
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
            nn.Conv2d(in_channels, channels // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, (channels // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d((channels // 4) * 3, channels, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=5, dilation=5, bias=False)
        )

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
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

    def forward(self, input: Tensor) -> Tensor:
        shortcut = self.shortcut(input)

        branch1 = self.branch1(input)
        branch2 = self.branch2(input)
        branch3 = self.branch3(input)
        branch4 = self.branch4(input)

        out = torch.cat((branch1, branch2, branch3, branch4), 1)
        out = self.conv1x1(out)

        out = out.mul(self.scale_ratio) + shortcut
        if self.lrelu is not None:
            out = self.lrelu(out)

        return out
