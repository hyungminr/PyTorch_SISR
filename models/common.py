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
