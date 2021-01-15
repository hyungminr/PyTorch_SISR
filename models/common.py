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
    def __init__(self):
        super(GMSD_quality, self).__init__()
        self.GMSD = GMSD()
        self.T = 170
    def forward(self, img1, img2):
        if img1.max() <= 1 and img2.max() <= 1:
            img1 = img1 * 255
            img2 = img2 * 255
        gm1 = self.GMSD(img1)
        gm2 = self.GMSD(img2)
        return 1 - (2 * gm1 * gm2 + self.T) / (torch.square(gm1) + torch.square(gm2) + self.T)
