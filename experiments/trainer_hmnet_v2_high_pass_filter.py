import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import datetime
import time
import numpy as np
import pandas as pd
import shutil
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import cv2

from collections import OrderedDict
from utils import imsave, sec2time, get_gpu_memory
from utils.eval import ssim as get_ssim
from utils.eval import ms_ssim as get_msssim
from utils.eval import psnr as get_psnr

from models.common import GMSD_quality
from models.common import MSHF
from models.common import Blur
from models.Morphology import Opening

from utils import pass_filter
from utils import high_pass_filter_hard_kernel

import warnings
warnings.simplefilter("ignore", UserWarning)

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

def get_hf_kernel(mode='high', sigma=2):
    kernel1d = cv2.getGaussianKernel(ksize=w, sigma=w * sigma / 100)
    kernel2d = np.outer(kernel1d, kernel1d.transpose())
    kernel2d = kernel2d / kernel2d.max()
    kernel2d = cv2.resize(kernel2d, dsize=(w, h))
    kernel2d = (kernel2d > 0.2) * 1.
    if mode == 'high': kernel2d = 1-kernel2d
    return kernel2d

quantize = lambda x: x.mul(255).clamp(0, 255).round().div(255)

def train(model, train_loader, test_loader, mode='EDSR_Baseline', save_image_every=50, save_model_every=10, test_model_every=1, epoch_start=0, num_epochs=1000, device=None, refresh=True, scale=2):

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    today = datetime.datetime.now().strftime('%Y.%m.%d')
    
    result_dir = f'./results/{today}/{mode}'
    weight_dir = f'./weights/{today}/{mode}'
    logger_dir = f'./logger/{today}_{mode}'
    csv = f'./hist_{today}_{mode}.csv'
    if refresh:
        try:
            shutil.rmtree(result_dir)
            shutil.rmtree(weight_dir)
            shutil.rmtree(logger_dir)
        except FileNotFoundError:
            pass
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(logger_dir, exist_ok=True)
    logger = SummaryWriter(log_dir=logger_dir, flush_secs=2)
    model = model.to(device)

    params = list(model.parameters())
    optim = torch.optim.Adam(params, lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1000, gamma= 0.99)
    criterion = torch.nn.L1Loss()
    
    start_time = time.time()
    print(f'Training Start || Mode: {mode}')

    step = 0
    pfix = OrderedDict()
    pfix_test = OrderedDict()

    hist = dict()
    hist['mode'] = f'{today}_{mode}'
    for key in ['epoch', 'psnr', 'ssim', 'ms-ssim']:
        hist[key] = []

    soft_mask = False
    
    # hf_kernel = get_hf_kernel(mode='high')

    for epoch in range(epoch_start, epoch_start+num_epochs):

        if epoch == 0:
            torch.save(model.state_dict(), f'{weight_dir}/epoch_{epoch+1:04d}.pth')
            
        if epoch == 0:
            with torch.no_grad():
                with tqdm(test_loader, desc=f'{mode} || Warming Up || Test Epoch {epoch}/{num_epochs}', position=0, leave=True) as pbar_test:
                    psnrs = []
                    ssims = []
                    msssims = []
                    for lr, lr_hf, hr, hr_hf, fname in pbar_test:
                        lr = lr.to(device)
                        hr = hr.to(device)
                        lr_hf = lr_hf.to(device)
                        
                        lhf = model(lr)
                        
                        lhf = quantize(lhf)
                        
                        psnr, ssim, msssim = evaluate(lr, lhf)
                        
                        psnrs.append(psnr)
                        ssims.append(ssim)
                        msssims.append(msssim)
                        
                        psnr_mean = np.array(psnrs).mean()
                        ssim_mean = np.array(ssims).mean()
                        msssim_mean = np.array(msssims).mean()

                        pfix_test['psnr'] = f'{psnr_mean:.4f}'
                        pfix_test['ssim'] = f'{ssim_mean:.4f}'
                        pfix_test['msssim'] = f'{msssim_mean:.4f}'

                        pbar_test.set_postfix(pfix_test)
                        if len(psnrs) > 1: break
                        

        with tqdm(train_loader, desc=f'{mode} || Epoch {epoch+1}/{num_epochs}', position=0, leave=True) as pbar:
            psnrs = []
            ssims = []
            msssims = []
            losses = []
            for lr, lr_hf, hr, hr_hf, _ in pbar:
            
                lr = lr.to(device)
                hr = hr.to(device)
                lr_hf = lr_hf.to(device)
                hr_hf = hr_hf.to(device)
                       
                # prediction
                lhf = model(lr)
                hhf = model(hr)
                
                loss_lr = criterion(lr_hf, lhf)
                loss_hr = criterion(hr_hf, hhf)
                
                # training
                loss_tot = loss_lr + loss_hr
                optim.zero_grad()
                loss_tot.backward()
                optim.step()
                scheduler.step()
                
                # training history 
                elapsed_time = time.time() - start_time
                elapsed = sec2time(elapsed_time)
                pfix['Loss LR'] = f'{loss_lr.item():.4f}'
                pfix['Loss HR'] = f'{loss_hr.item():.4f}'
                
                lhf = quantize(lhf)
                psnr, ssim, _ = evaluate(lr_hf, lhf)
                psnrs.append(psnr)
                ssims.append(ssim)
                
                hhf = quantize(hhf)
                psnr, ssim, _ = evaluate(hr_hf, hhf)
                psnrs.append(psnr)
                ssims.append(ssim)
                
                psnr_mean = np.array(psnrs).mean()
                ssim_mean = np.array(ssims).mean()

                pfix['PSNR'] = f'{psnr_mean:.2f}'
                pfix['SSIM'] = f'{ssim_mean:.4f}'
                           
                free_gpu = get_gpu_memory()[0]
                
                pfix['free GPU'] = f'{free_gpu}MiB'
                pfix['Elapsed'] = f'{elapsed}'
                
                pbar.set_postfix(pfix)
                losses.append(loss_tot.item())
                

                    
                step += 1
                
            logger.add_scalar("Loss/train", np.array(losses).mean(), epoch+1)
            logger.add_scalar("PSNR/train", psnr_mean, epoch+1)
            logger.add_scalar("SSIM/train", ssim_mean, epoch+1)
            
            if (epoch+1) % save_model_every == 0:
                torch.save(model.state_dict(), f'{weight_dir}/epoch_{epoch+1:04d}.pth')
                
            if (epoch+1) % test_model_every == 0:
                
                with torch.no_grad():
                    with tqdm(test_loader, desc=f'{mode} || Test Epoch {epoch+1}/{num_epochs}', position=0, leave=True) as pbar_test:
                        psnrs = []
                        ssims = []
                        msssims = []
                        for lr, lr_hf, hr, hr_hf, fname in pbar_test:
                            
                            fname = fname[0].split('/')[-1].split('.pt')[0]
                            
                            lr = lr.to(device)
                            hr = hr.to(device)
                            lr_hf = lr_hf.to(device)
                            hr_hf = hr_hf.to(device)
                            
                            lhf = model(lr)
                            hhf = model(hr)
                                                        
                            lhf = quantize(lhf)
                            psnr, ssim, msssims = evaluate(lr_hf, lhf)
                            psnrs.append(psnr)
                            ssims.append(ssim)
                            
                            hhf = quantize(hhf)
                            psnr, ssim, msssims = evaluate(hr_hf, hhf)
                            psnrs.append(psnr)
                            ssims.append(ssim)
                            
                            psnr_mean = np.array(psnrs).mean()
                            ssim_mean = np.array(ssims).mean()
                            msssim_mean = np.array(msssims).mean()

                            psnr_mean = np.array(psnrs).mean()
                            ssim_mean = np.array(ssims).mean()
                            msssim_mean = np.array(msssims).mean()

                            pfix_test['psnr'] = f'{psnr_mean:.4f}'
                            pfix_test['ssim'] = f'{ssim_mean:.4f}'
                            pfix_test['msssim'] = f'{msssim_mean:.4f}'
                            
                            pbar_test.set_postfix(pfix_test)
                            
                            z = torch.zeros_like(lr[0])
                            
                            xz = torch.cat([lr[0], lhf[0], lr_hf[0], z], dim=-2)
                            imsave([xz, hr[0], hhf[0], hr_hf[0]], f'{result_dir}/{fname}.jpg')
                            
                            
                            
                        hist['epoch'].append(epoch+1)
                        hist['psnr'].append(psnr_mean)
                        hist['ssim'].append(ssim_mean)
                        hist['ms-ssim'].append(msssim_mean)
                        
                        logger.add_scalar("PSNR/test", psnr_mean, epoch+1)
                        logger.add_scalar("SSIM/test", ssim_mean, epoch+1)
                        logger.add_scalar("MS-SSIM/test", msssim_mean, epoch+1)
                        
                        df = pd.DataFrame(hist)
                        df.to_csv(csv)
