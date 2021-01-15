import os
import time
import numpy as np
import subprocess
import torch
import torchvision.transforms as T
from PIL import Image

def get_gpu_memory():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(subprocess.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    # print(memory_free_values)
    return memory_free_values

def sec2time(sec, n_msec=0):
    if hasattr(sec,'__len__'): return [sec2time(s) for s in sec]    
    m, s = divmod(sec, 60)    
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if n_msec > 0: pattern = '%%02d:%%02d:%%0%d.%df' % (n_msec+3, n_msec)
    else: pattern = r'%02d:%02d:%02d'
    if d == 0: return pattern % (h, m, s)
    return ('%d days, ' + pattern) % (d, h, m, s)
    
def imshow(tensor, resize=None, visualize=True):
    if type(tensor) is list:
        pad = torch.nn.ConstantPad2d(2, 0)
        for i, t in enumerate(tensor):
            tensor[i] = pad(t)
            if len(tensor[i].shape) == 4:
                tensor[i] = tensor[i][0]
        tensor = torch.cat(tensor, dim=-1) # horizontal concat
    t2img = T.ToPILImage()
    # img = t2img(torch.clamp(tensor[0].cpu().detach(), 0, 1))
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    img = t2img(torch.clamp(tensor, 0, 1))
    if resize is not None:
        img = img.resize(resize)
    if visualize:
        display(img)
        return
    else:
        return img
    
def imsave(tensor, filename, resize=None):
    if type(tensor) is list:
        pad = torch.nn.ConstantPad2d(2, 0)
        for i, t in enumerate(tensor):
            tensor[i] = pad(t)
        tensor = torch.cat(tensor, dim=-1) # horizontal concat
    t2img = T.ToPILImage()
    # img = t2img(torch.clamp(tensor[0].cpu().detach(), 0, 1))
    img = t2img(torch.clamp(tensor, 0, 1))
    if resize is not None:
        img = img.resize(resize)
    img.save(filename)
    return
    
def normalize(tensor, m=None, M=None):
    m = tensor.min() if m is None else m
    M = tensor.max() if M is None else M
    return (tensor-m)/(M-m)
    
def get_gaussian_kernel(sigma=2.5, ksize=17):
    #  generate a (2k+1)x(2k+1) gaussian kernel with mean=0 and sigma = s
    kd2 = (ksize-1)//2
    probs = [np.exp(-z*z/(2*sigma*sigma))/np.sqrt(2*np.pi*sigma*sigma) for z in range(-kd2,kd2+1)] 
    kernel = np.outer(probs, probs)
    kernel = torch.tensor(kernel, dtype=torch.float32)
    kernel = kernel / kernel.sum()
    return kernel

def get_aniso_gaussian_kernel(sigma=2.5, ksize=17, degree=90):
        
    t2img = T.ToPILImage()
    rotation = T.Compose([T.RandomRotation(degrees=(degree, degree)), T.ToTensor()])
    
    img2tensor = T.Compose([T.ToTensor()])
    
    km10 = ksize * 2 + 1
    kernel_iso = get_gaussian_kernel(sigma=sigma, ksize=ksize)
    kernel_iso = img2tensor(t2img(kernel_iso).resize((km10, km10))).squeeze(0)
    km10d2 = (km10-1)//2
        
    z = torch.zeros((km10d2, km10))
    kernel_wide = torch.cat((z, kernel_iso, z), dim=0)
    kernel_rotate = rotation(t2img(kernel_wide).resize((km10, km10)))
    image = t2img(kernel_rotate).resize((ksize, ksize))
    kernel = img2tensor(image)
    kernel = kernel.squeeze(0)/kernel.sum()
    return kernel
