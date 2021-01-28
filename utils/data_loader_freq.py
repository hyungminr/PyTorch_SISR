import os
import glob
import numpy as np
import math
import random
import torch
import torchvision.transforms as T
from utils import evaluate

class dataset(torch.utils.data.Dataset):
    """ Load HR / LR pair """
    def __init__(self, data='DIV2K', mode='test', height=96, width=96, scale_factor=2, augment=False):
        if data == 'DIV2K':
            if mode == 'test':
                self.root_dir = './data/DIV2K/bin/DIV2K_valid_HR/'
            else:
                self.root_dir = './data/DIV2K/bin/DIV2K_train_HR/'
        self.height = 256 if mode=='test' else height
        self.width = 256 if mode=='test' else width
        self.augment = augment
        self.files = self.find_files()
        self.scale_factor = scale_factor
        self.up = torch.nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False)
    
    def find_files(self):
        return glob.glob(f'{self.root_dir}/*.pt')
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        
        hflip = random.choice([True, False]) if self.augment else False
        vflip = random.choice([True, False]) if self.augment else False
        
        index = self.indexerror(index)
        
        output_name = self.files[index]
        input_name = output_name.replace('HR', f'LR_bicubic/X{self.scale_factor}')
        input_name = input_name.replace('.pt', f'x{self.scale_factor}.pt')
        input_hf_name = input_name.replace('/bin/', '/bin/hfreq/')
       
               
        input_tensor = torch.load(input_name)
        input_hf_tensor = torch.load(input_hf_name)
        output_tensor = torch.load(output_name)
        
        if self.height > 0 and self.width > 0:
            
            crop = self.get_crop_bbox(input_tensor)
            input_tensor_cropped = self.crop_image(input_tensor, crop, mode='lr')
            input_hf_tensor_cropped = self.crop_image(input_hf_tensor, crop, mode='lr')
            output_tensor_cropped = self.crop_image(output_tensor, crop, mode='hr')
            
            if self.augment:
                psnr, ssim, _ = evaluate(self.up(input_tensor_cropped.unsqueeze(0)), output_tensor_cropped.unsqueeze(0))
                cnt = 0
                while psnr > 40 or ssim > 0.95:
                    cnt += 1
                    crop = self.get_crop_bbox(input_tensor)
                    input_tensor_cropped = self.crop_image(input_tensor, crop, mode='lr')
                    input_hf_tensor_cropped = self.crop_image(input_hf_tensor, crop, mode='lr')
                    output_tensor_cropped = self.crop_image(output_tensor, crop, mode='hr')
                    psnr, ssim, _ = evaluate(self.up(input_tensor_cropped.unsqueeze(0)), output_tensor_cropped.unsqueeze(0))
                    if cnt > 10: break
                    
            input_tensor = input_tensor_cropped
            input_hf_tensor = input_hf_tensor_cropped
            output_tensor = output_tensor_cropped
                    
        if hflip:
            input_tensor= torch.tensor(input_tensor.numpy()[:,:,::-1].copy())
            input_hf_tensor= torch.tensor(input_hf_tensor.numpy()[:,:,::-1].copy())
            output_tensor= torch.tensor(output_tensor.numpy()[:,:,::-1].copy())
            
        if vflip:
            input_tensor= torch.tensor(input_tensor.numpy()[:,::-1,:].copy())
            input_hf_tensor= torch.tensor(input_hf_tensor.numpy()[:,::-1,:].copy())
            output_tensor= torch.tensor(output_tensor.numpy()[:,::-1,:].copy())
            
        return input_tensor, input_hf_tensor, output_tensor, output_name

    def get_crop_bbox(self, tensor):
        _, tensor_width, tensor_height = tensor.shape
        w = self.width // self.scale_factor
        h = self.height // self.scale_factor
        if self.augment:
            left = np.random.randint(max(1, tensor_width - w - 1))
            top = np.random.randint(max(1, tensor_height - h - 1))
        else:
            left = (tensor_width - w) // 2
            top = (tensor_height - h) // 2
        return left, top
        
    def crop_image(self, tensor, crop_shape, mode):
        w = self.width // self.scale_factor
        h = self.height // self.scale_factor
        _, width, height = tensor.shape
        if mode == 'lr':
            crop_shape = [i for i in crop_shape]
        elif mode == 'hr':
            crop_shape = [i * self.scale_factor for i in crop_shape]
            w = w * self.scale_factor
            h = h * self.scale_factor
        return tensor[:,crop_shape[0]:crop_shape[0]+w,crop_shape[1]:crop_shape[1]+h]
    
    def indexerror(self, index):
        index = index if index < len(self.files) else 0
        return index
    

def get_loader(data='DIV2K', mode='test', batch_size=1, num_workers=1, height=96, width=96, scale_factor=2, augment=False):
        
    shuffle = (mode == 'train')    
    data_loader = torch.utils.data.DataLoader(dataset=dataset(data, mode, height, width, scale_factor, augment),
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              prefetch_factor=10)   
    return data_loader

