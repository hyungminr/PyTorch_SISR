import os
import time
import numpy as np
import subprocess
import torch
import torchvision.transforms as T
from PIL import Image
import cv2
from utils.eval import ssim as get_ssim
from utils.eval import ms_ssim as get_msssim
from utils.eval import psnr as get_psnr

def img2tensor(iname, device='cuda'):
    trans = T.Compose([T.ToTensor()])
    img = Image.open(iname)
    return trans(img).unsqueeze(0).to(device)
    

def evaluate(hr: torch.tensor, sr: torch.tensor):
    batch_size, _, h, w = hr.shape
    psnrs, ssims, msssims = [], [], []
    for b in range(batch_size):
        psnrs.append(get_psnr(hr[b], sr[b]))
        ssims.append(get_ssim(hr[b].unsqueeze(0), sr[b].unsqueeze(0)).item())
        if h > 160 and w > 160:
            msssim = get_msssim(hr[b].unsqueeze(0), sr[b].unsqueeze(0)).item()
        else:
            msssim = 0
        msssims.append(msssim)    
    return np.array(psnrs).mean(), np.array(ssims).mean(), np.array(msssims).mean()


def fea2img(x):
    pad = torch.nn.ConstantPad2d(1, 1)
    trans = [T.ToTensor()]
    trans = T.Compose(trans)
    t2img = T.ToPILImage()
    x = x[0].cpu()
    d, h, w = x.shape
    if d == 12: row, col = 2, 6
    elif d == 64: row, col = 8, 8
    else: row, col = 1, d
    xxx = []
    for i in range(row):
        xx = []
        for j in range(col):
            xz = x[col*i+j]
            xz = (xz - xz.min()) / (xz.max() - xz.min())
            xz = pad(xz.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            xx.append(xz)
        xx = torch.cat(xx, dim=-1)
        xxx.append(xx)
    x = torch.cat(xxx, dim=-2)
    return t2img(x)
        
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
    
def imshow(tensor, resize=None, visualize=True, filename=None):
    if type(tensor) is list:
        pad = torch.nn.ConstantPad2d(1, 1)
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
    if filename is not None:
        img.save(filename)
    if visualize:
        display(img)
        return
    return img
        
def imsave(tensor, filename, resize=None):
    imshow(tensor, resize=resize, visualize=False, filename=filename)
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
    
    
def high_pass_filter(tensor_input, sigma=50):
    tensor_output = torch.zeros_like(tensor_input)
    tensor_input = torch.transpose(tensor_input, 2, 3)
    tensor_input = torch.transpose(tensor_input, 1, 3)
    b, h, w, _ = tensor_input.shape
    for bi in range(b):
        tensor = tensor_input[bi, :, :, :]
        tensor = tensor.view(h, w, 3)
        img = tensor.cpu().numpy()
        img = np.array(img * 255, dtype=np.uint8)
        img_high = np.zeros_like(img)
        img_low = np.zeros_like(img)
        h, w, _ = img.shape
        for i in [0, 1, 2]:
            grey = img[:,:,i] # gray-scale image
            # r = 50 # how narrower the window is
            # ham = np.hamming(h)[:,None] # 1D hamming
            # ham2d = np.sqrt(np.dot(ham, ham.T)) ** r # expand to 2D hamming

            kernel1d = cv2.getGaussianKernel(ksize=w, sigma=sigma)
            kernel2d = np.outer(kernel1d, kernel1d.transpose())
            kernel2d = kernel2d / kernel2d.max()
            kernel2d = cv2.resize(kernel2d, dsize=(w, h))
            f = cv2.dft(grey.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
            f_shifted = np.fft.fftshift(f)
            f_complex = f_shifted[:,:,0]*1j + f_shifted[:,:,1]

            f_filtered = (1-kernel2d) * f_complex
            f_filtered_shifted = np.fft.fftshift(f_filtered)
            inv_img = np.fft.ifft2(f_filtered_shifted) # inverse F.T.
            filtered_img = np.abs(inv_img)
            filtered_img -= filtered_img.min()
            filtered_img = filtered_img / (filtered_img.max() + 1e-7)
            filtered_img = filtered_img * 255
            filtered_img = filtered_img.astype(np.uint8)

            img_high[:,:,i] = filtered_img
        res = torch.tensor(img_high)
        res = torch.transpose(res, 0, 2)
        res = torch.transpose(res, 1, 2)
        tensor_output[bi, :, :, :] = res.unsqueeze(0)
    return tensor_output / 255
    
    

def pass_filter(tensor_input, sigma=20):
    tensor_high = torch.zeros_like(tensor_input)
    tensor_low = torch.zeros_like(tensor_input)
    tensor_input = torch.transpose(tensor_input, 2, 3)
    tensor_input = torch.transpose(tensor_input, 1, 3)
    b, h, w, _ = tensor_input.shape
    for bi in range(b):
        tensor = tensor_input[bi, :, :, :]
        tensor = tensor.view(h, w, 3)
        img = tensor.cpu().numpy()
        img = np.array(img * 255, dtype=np.uint8)
        img_high = np.zeros_like(img)
        img_low = np.zeros_like(img)
        h, w, _ = img.shape
        for i in [0, 1, 2]:
            grey = img[:,:,i] # gray-scale image
            # r = 50 # how narrower the window is
            # ham = np.hamming(h)[:,None] # 1D hamming
            # ham2d = np.sqrt(np.dot(ham, ham.T)) ** r # expand to 2D hamming

            kernel1d = cv2.getGaussianKernel(ksize=w, sigma=sigma)
            kernel2d = np.outer(kernel1d, kernel1d.transpose())
            kernel2d = kernel2d / kernel2d.max()
            kernel2d = cv2.resize(kernel2d, dsize=(w, h))
            
            f = cv2.dft(grey.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
            f_shifted = np.fft.fftshift(f)
            f_complex = f_shifted[:,:,0]*1j + f_shifted[:,:,1]

            # high pass filter

            f_filtered = (1-kernel2d) * f_complex
            f_filtered_shifted = np.fft.fftshift(f_filtered)
            inv_img = np.fft.ifft2(f_filtered_shifted) # inverse F.T.
            filtered_img = np.abs(inv_img)
            filtered_img -= filtered_img.min()
            filtered_img = filtered_img * 255 / (filtered_img.max() + 1e-7)
            filtered_img = filtered_img.astype(np.uint8)
            img_high[:,:,i] = filtered_img
            
            # low pass filter

            f_filtered = kernel2d * f_complex
            f_filtered_shifted = np.fft.fftshift(f_filtered)
            inv_img = np.fft.ifft2(f_filtered_shifted) # inverse F.T.
            filtered_img = np.abs(inv_img)
            filtered_img -= filtered_img.min()
            filtered_img = filtered_img * 255 / (filtered_img.max() + 1e-7)
            filtered_img = filtered_img.astype(np.uint8)
            img_low[:,:,i] = filtered_img
            
            
        res = torch.tensor(img_high)
        res = torch.transpose(res, 0, 2)
        res = torch.transpose(res, 1, 2)
        tensor_high[bi, :, :, :] = res.unsqueeze(0)
        
        res = torch.tensor(img_low)
        res = torch.transpose(res, 0, 2)
        res = torch.transpose(res, 1, 2)
        tensor_low[bi, :, :, :] = res.unsqueeze(0)
        
    return tensor_high / 255, tensor_low / 255
    
def high_pass_filter_hard_kernel(tensor_input, sigma=2, mode='high', kernel = None):
    tensor_output = torch.zeros_like(tensor_input)
    tensor_input = torch.transpose(tensor_input, 2, 3)
    tensor_input = torch.transpose(tensor_input, 1, 3)
    b, h, w, _ = tensor_input.shape
    for bi in range(b):
        tensor = tensor_input[bi, :, :, :]
        tensor = tensor.view(h, w, 3)
        img = tensor.cpu().numpy()
        img = np.array(img * 255, dtype=np.uint8)
        img_high = np.zeros_like(img)
        img_low = np.zeros_like(img)
        h, w, _ = img.shape
        for i in [0, 1, 2]:
            grey = img[:,:,i] # gray-scale image
            # r = 50 # how narrower the window is
            # ham = np.hamming(h)[:,None] # 1D hamming
            # ham2d = np.sqrt(np.dot(ham, ham.T)) ** r # expand to 2D hamming
            
            if kernel is None:
                
                kernel1d = cv2.getGaussianKernel(ksize=w, sigma=w * sigma / 100)
                kernel2d = np.outer(kernel1d, kernel1d.transpose())
                kernel2d = kernel2d / kernel2d.max()
                kernel2d = cv2.resize(kernel2d, dsize=(w, h))
                kernel2d = (kernel2d > 0.2) * 1.
                if mode == 'high': kernel2d = 1-kernel2d
            else:
                kernel2d = kernel
            
            f = cv2.dft(grey.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
            f_shifted = np.fft.fftshift(f)
            f_complex = f_shifted[:,:,0]*1j + f_shifted[:,:,1]

            f_filtered = kernel2d * f_complex
            f_filtered_shifted = np.fft.fftshift(f_filtered)
            inv_img = np.fft.ifft2(f_filtered_shifted) # inverse F.T.
            filtered_img = np.abs(inv_img)
            filtered_img -= filtered_img.min()
            filtered_img = filtered_img * 255 / (filtered_img.max() + 1e-7)
            filtered_img = filtered_img.astype(np.uint8)

            img_high[:,:,i] = filtered_img
        res = torch.tensor(img_high)
        res = torch.transpose(res, 0, 2)
        res = torch.transpose(res, 1, 2)
        tensor_output[bi, :, :, :] = res.unsqueeze(0)
    return tensor_output / 255
