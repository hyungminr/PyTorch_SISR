import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.RCAN import RCAN
from models.RCAN import MeanShift
    
class RCAN_freq(nn.Module):
    """  """
    def __init__(self, num_RG=10, scale=2, kernel=3, padding=1, bias=True, num_feats=64):
        super().__init__()
        
        self.sub_mean = MeanShift(mode='sub')
        
        self.model_image = RCAN(scale=scale, num_feats=32)
        self.model_high  = RCAN(scale=scale, num_feats=24)
        self.model_low   = RCAN(scale=scale, num_feats= 8)
        
        layers = []
        if (scale & (scale - 1)) == 0: # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                layers += [nn.Conv2d(in_channels=num_feats, out_channels=num_feats*4, kernel_size=kernel, padding=padding, bias=bias)]
                layers += [nn.PixelShuffle(2)]
        layers += [nn.Conv2d(in_channels=num_feats, out_channels=3, kernel_size=kernel, padding=padding, bias=bias)]
        self.tail = nn.Sequential(*layers)        
        self.add_mean = MeanShift(mode='add')
        
    def forward(self, img, high, low):
        sr_image, deep_image = self.model_image(img)
        sr_high, deep_high = self.model_high(high)
        sr_low, deep_low = self.model_low(low)
        deep = torch.cat((deep_image[0] + deep_image[-1],
                          deep_high[0] + deep_high[-1],
                          deep_low[0] + deep_low[-1]), dim=1)
        x_up = self.tail(deep)
        out = self.add_mean(x_up)
        return out, [sr_image, sr_high, sr_low]
