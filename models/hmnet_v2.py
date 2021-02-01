import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn

class ReLU1(torch.nn.modules.activation.Hardtanh):
    def __init__(self, inplace: bool = False):
        super(ReLU1, self).__init__(0., 1., inplace)
    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
        
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
        outs = []
        hh = self.scale_3[0](x)
        vv = self.scale_3[1](x)
        hv = self.scale_3[2](x)        
        
        fea = torch.square(hh - vv) + 4 * torch.square(hv)
        fea = torch.sqrt(fea + self.eps)        
        fea_3 = (hh + vv + fea) * 0.5
        fea_3 = torch.mean(fea_3, dim=1, keepdim=True)
        
        hh = torch.mean(hh, dim=1, keepdim=True)
        vv = torch.mean(vv, dim=1, keepdim=True)
        hv = torch.mean(hv, dim=1, keepdim=True)
        outs.append(hh)        
        outs.append(vv)        
        outs.append(hv)
        
        hh = self.scale_5[0](x)
        vv = self.scale_5[1](x)
        hv = self.scale_5[2](x)     
        
        fea = torch.square(hh - vv) + 4 * torch.square(hv)
        fea = torch.sqrt(fea + self.eps)        
        fea_5 = (hh + vv + fea) * 0.5
        fea_5 = torch.mean(fea_5, dim=1, keepdim=True)
        
        hh = torch.mean(hh, dim=1, keepdim=True)
        vv = torch.mean(vv, dim=1, keepdim=True)
        hv = torch.mean(hv, dim=1, keepdim=True)
        outs.append(hh)        
        outs.append(vv)        
        outs.append(hv)
        
        hh = self.scale_7[0](x)
        vv = self.scale_7[1](x)
        hv = self.scale_7[2](x)     
        
        fea = torch.square(hh - vv) + 4 * torch.square(hv)
        fea = torch.sqrt(fea + self.eps)        
        fea_7 = (hh + vv + fea) * 0.5
        fea_7 = torch.mean(fea_7, dim=1, keepdim=True)
        
        hh = torch.mean(hh, dim=1, keepdim=True)
        vv = torch.mean(vv, dim=1, keepdim=True)
        hv = torch.mean(hv, dim=1, keepdim=True)
        outs.append(hh)        
        outs.append(vv)        
        outs.append(hv)
        
        outs += [fea_3, fea_5, fea_7]
        
        out = torch.cat(outs, dim=1)
        out = self.act(out * (self.rgb_scale / 255))
        return out

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
    def __init__(self, kernel=3, num_feats=64, padding=1, bias=True, res_scale=1):
        super().__init__()
        
        layers = []
        layers += [nn.Conv2d(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel, padding=padding, bias=bias)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel, padding=padding, bias=bias)]
        self.conv = nn.Sequential(*layers)
        self.res_scale = res_scale
        
    def forward(self, x):
        res = self.conv(x).mul(self.res_scale)
        return x + res
        

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
    def __init__(self, num_RCAB=2, kernel=3, padding=1, bias=True, num_feats=64):
        super().__init__()
        
        layers = []
        layers += [RCAB(num_feats=num_feats) for _ in range(num_RCAB)]
        layers += [nn.Conv2d(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel, padding=padding, bias=bias)]
        self.rcab = nn.Sequential(*layers)
        
    def forward(self, x):
        return x + self.rcab(x)
        
class ChannAtt(nn.Module):
    def __init__(self):
        super().__init__()
        
        layers = []
        layers += [nn.AdaptiveAvgPool2d(1)]
        layers += [nn.Conv2d(in_channels=64, out_channels= 4, kernel_size=3, padding=1, bias=True)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(in_channels= 4, out_channels=64, kernel_size=3, padding=1, bias=True)]
        layers += [nn.Sigmoid()]
        self.channel_att = nn.Sequential(*layers)
        
    def forward(self, x_fea):
        x_cha = self.channel_att(x_fea)
        x_att = x_fea * x_cha
        return x_att

                
class FeatureAtt2(nn.Module):
    def __init__(self):
        super().__init__()
        self.att = ChannAtt()
        self.pool = nn.AdaptiveAvgPool2d(1)
        layers = [nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), padding=0, bias=True)]
        layers += [nn.Softmax(dim=2)]
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x, y):
        px = self.pool(self.att(x))
        py = self.pool(self.att(y))
        pf = torch.cat([px, py, px, py], dim=2)
        pw = self.conv(pf) * 2
        fea_map = torch.cat([x.unsqueeze(2), y.unsqueeze(2)], dim=2) * pw.unsqueeze(-1)
        return torch.sum(fea_map, dim=2)
                
class FeatureAtt3(nn.Module):
    def __init__(self):
        super().__init__()
        self.att0 = ChannAtt()
        self.att1 = ChannAtt()
        self.att2 = ChannAtt()
        self.pool = nn.AdaptiveAvgPool2d(1)
        layers = [nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), padding=0, bias=True)]
        layers += [nn.Softmax(dim=2)]
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x, y, z):
        px = self.pool(self.att0(x))
        py = self.pool(self.att1(y))
        pz = self.pool(self.att2(z))
        pf = torch.cat([px, py, pz, px, py], dim=2)
        pw = self.conv(pf) * 3
        fea_map = torch.cat([x.unsqueeze(2), y.unsqueeze(2), z.unsqueeze(2)], dim=2) * pw.unsqueeze(-1)
        return torch.sum(fea_map, dim=2)
        
class FeatureAtt4(nn.Module):
    def __init__(self):
        super().__init__()
        self.att0 = ChannAtt()
        self.att1 = ChannAtt()
        self.att2 = ChannAtt()
        self.att3 = ChannAtt()
        self.pool = nn.AdaptiveAvgPool2d(1)
        layers = [nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 1), padding=0, bias=True)]
        layers += [nn.Softmax(dim=2)]
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x, y, z, a):
        px = self.pool(self.att0(x))
        py = self.pool(self.att1(y))
        pz = self.pool(self.att2(z))
        pa = self.pool(self.att3(a))
        pf = torch.cat([px, py, pz, pa, px, py, pz], dim=2)
        pw = self.conv(pf) * 4
        fea_map = torch.cat([x.unsqueeze(2), y.unsqueeze(2), z.unsqueeze(2), a.unsqueeze(2)], dim=2) * pw.unsqueeze(-1)
        return torch.sum(fea_map, dim=2)
        
class FeatureAtt5(nn.Module):
    def __init__(self):
        super().__init__()
        self.att0 = ChannAtt()
        self.att1 = ChannAtt()
        self.att2 = ChannAtt()
        self.att3 = ChannAtt()
        self.att4 = ChannAtt()
        self.pool = nn.AdaptiveAvgPool2d(1)
        layers = [nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 1), padding=0, bias=True)]
        layers += [nn.Softmax(dim=2)]
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x, y, z, a, b):
        px = self.pool(self.att0(x))
        py = self.pool(self.att1(y))
        pz = self.pool(self.att2(z))
        pa = self.pool(self.att3(a))
        pb = self.pool(self.att4(b))
        pf = torch.cat([pa, pb, px, py, pz, pa, pb, px, py], dim=2)
        pw = self.conv(pf) * 5
        fea_map = torch.cat([x.unsqueeze(2), y.unsqueeze(2), z.unsqueeze(2), a.unsqueeze(2), b.unsqueeze(2)], dim=2) * pw.unsqueeze(-1)
        return torch.sum(fea_map, dim=2)
                
class hmnet(nn.Module):
    """  """
    def __init__(self, args=None, num_RB=16, num_feats=64, scale=4, kernel=3, padding=1, bias=True):
        super().__init__()
                
        self.upx2_bicubic = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        self.upx4_bicubic = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)
                
        self.sub_mean = MeanShift(mode='sub')
        
        layers =  [nn.Conv2d(in_channels=3, out_channels=num_feats, kernel_size=kernel, padding=padding, bias=bias)]
        layers += [nn.Conv2d(in_channels=3, out_channels=num_feats, kernel_size=kernel, padding=padding, bias=bias)]
        layers += [nn.Conv2d(in_channels=3, out_channels=num_feats, kernel_size=kernel, padding=padding, bias=bias)]
        self.heads = nn.ModuleList(layers)
                
        blocks  = [RG(num_RCAB= 2, num_feats=num_feats, kernel=kernel, padding=padding, bias=bias)]
        blocks += [RG(num_RCAB= 8, num_feats=num_feats, kernel=kernel, padding=padding, bias=bias)]
        blocks += [RG(num_RCAB=32, num_feats=num_feats, kernel=kernel, padding=padding, bias=bias)]
        self.bodies = nn.ModuleList(blocks)
                
        layers  = [nn.Conv2d(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel, padding=padding, bias=bias, stride=2)]
        layers += [nn.Conv2d(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel, padding=padding, bias=bias, stride=2)]
        self.downConv = nn.ModuleList(layers)
                
        layers  = [nn.Conv2d(in_channels=num_feats, out_channels=num_feats*4, kernel_size=kernel, padding=padding, bias=bias)]
        layers += [nn.Conv2d(in_channels=num_feats, out_channels=num_feats*4, kernel_size=kernel, padding=padding, bias=bias)]
        self.upConv = nn.ModuleList(layers)
        
        self.pixelShuffle = nn.PixelShuffle(2)
                
        layers  = [nn.Conv2d(in_channels=num_feats, out_channels=3, kernel_size=kernel, padding=padding, bias=bias)]
        layers += [nn.Conv2d(in_channels=num_feats, out_channels=3, kernel_size=kernel, padding=padding, bias=bias)]
        layers += [nn.Conv2d(in_channels=num_feats, out_channels=3, kernel_size=kernel, padding=padding, bias=bias)]
        self.tails = nn.ModuleList(layers)
        
        self.add_mean = MeanShift(mode='add')
        
        self.mshf = MSHF(3, 3)
        layers  = [nn.Conv2d(in_channels=12, out_channels=num_feats, kernel_size=1, padding=0, bias=bias)]
        layers += [nn.Conv2d(in_channels=12, out_channels=num_feats, kernel_size=1, padding=0, bias=bias)]
        layers += [nn.Conv2d(in_channels=12, out_channels=num_feats, kernel_size=1, padding=0, bias=bias)]
        self.mshf_tail = nn.ModuleList(layers)
        
        self.feaAtt_x1_down = FeatureAtt2()
        self.feaAtt_x2_down = FeatureAtt2()
        
        self.feaAtt_x1 = FeatureAtt3()
        self.feaAtt_x2 = FeatureAtt4()
        self.feaAtt_x4 = FeatureAtt4()
        
    def forward(self, img):    

        # preprocess: meanshift & upscale
        imgx1 = self.sub_mean(img)
        imgx2 = self.upx2_bicubic(imgx1)
        imgx4 = self.upx4_bicubic(imgx1)
        
        # mshf feature
        fea_mshf_x4 = self.mshf_tail[0](self.mshf(imgx4))
        fea_mshf_x2 = self.mshf_tail[1](self.mshf(imgx2))
        fea_mshf_x1 = self.mshf_tail[2](self.mshf(imgx1))
        
        # shallow feature
        fea_shallow_x4 = self.heads[0](imgx4)
        fea_shallow_x2 = self.heads[1](imgx2)
        fea_shallow_x1 = self.heads[2](imgx1)
        
        fea_shallow_x2 = self.feaAtt_x2_down(fea_shallow_x2, self.downConv[0](fea_shallow_x4))
        fea_shallow_x1 = self.feaAtt_x1_down(fea_shallow_x1, self.downConv[1](fea_shallow_x2))
        
        # deep feature
                
        fea_deep_x4 = self.bodies[0](fea_shallow_x4)
        fea_deep_x2 = self.bodies[1](fea_shallow_x2)
        fea_deep_x1 = self.bodies[2](fea_shallow_x1)
        
        # shallow feature + deep feature
        ## fea_x1 = fea_shallow_x1 + fea_deep_x1 + fea_mshf_x1
        fea_x1 = self.feaAtt_x1(fea_shallow_x1, fea_deep_x1, fea_mshf_x1)
        fea_x1_to_x2 = self.pixelShuffle(self.upConv[0](fea_x1))
        
        ## fea_x2 = fea_shallow_x2 + fea_deep_x2 + fea_x1_to_x2 + fea_mshf_x2
        fea_x2 = self.feaAtt_x2(fea_shallow_x2, fea_deep_x2, fea_x1_to_x2, fea_mshf_x2)
        fea_x2_to_x4 = self.pixelShuffle(self.upConv[1](fea_x2))
        
        ## fea_x4 = fea_shallow_x4 + fea_deep_x4 + fea_x2_to_x4 + fea_mshf_x4
        fea_x4 = self.feaAtt_x4(fea_shallow_x4, fea_deep_x4, fea_x2_to_x4, fea_mshf_x4)
        
        # x_feature = x_deep[0] + x_deep[-2] + x_deep[-1]
        # x_feature = self.feature_attention(x_deep[0], x_deep[-1], x_mshf, x_deep_hf[0], x_deep_hf[-1])
        
        # deblur
        up_x4 = self.tails[0](fea_x4)
        up_x2 = self.tails[1](fea_x2)
        up_x1 = self.tails[2](fea_x1)
        
        # postprocess: meanshift
        img_x4 = self.add_mean(up_x4)
        img_x2 = self.add_mean(up_x2)
        img_x1 = self.add_mean(up_x1)
        
        return img_x4, img_x2, img_x1
