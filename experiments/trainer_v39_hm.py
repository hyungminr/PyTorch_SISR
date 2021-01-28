import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
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
    GMSD = GMSD_quality().to(device)
    opening = Opening().to(device)
    blur = Blur().to(device)
    mshf = MSHF(3, 3).to(device)
    
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
                    for lr, lr_hf, hr, fname in pbar_test:
                        lr = lr.to(device)
                        hr = hr.to(device)
                        lr_hf = lr_hf.to(device)
                                        
                        
                        sr, deep = model(lr, lr_hf)
                        
                        sr = quantize(sr)
                        
                        psnr, ssim, msssim = evaluate(hr, sr)
                        
                        psnrs.append(psnr)
                        ssims.append(ssim)
                        msssims.append(msssim)
                        
                        psnr_mean = np.array(psnrs).mean()
                        ssim_mean = np.array(ssims).mean()
                        msssim_mean = np.array(msssims).mean()

                        pfix_test['psnr'] = f'{psnr:.4f}'
                        pfix_test['ssim'] = f'{ssim:.4f}'
                        pfix_test['msssim'] = f'{msssim:.4f}'
                        pfix_test['psnr_mean'] = f'{psnr_mean:.4f}'
                        pfix_test['ssim_mean'] = f'{ssim_mean:.4f}'
                        pfix_test['msssim_mean'] = f'{msssim_mean:.4f}'

                        pbar_test.set_postfix(pfix_test)
                        if len(psnrs) > 1: break
                        

        with tqdm(train_loader, desc=f'{mode} || Epoch {epoch+1}/{num_epochs}', position=0, leave=True) as pbar:
            psnrs = []
            ssims = []
            msssims = []
            losses = []
            for lr, lr_hf, hr, _ in pbar:
                lr = lr.to(device)
                hr = hr.to(device)
                lr_hf = lr_hf.to(device)
                                
                # prediction
                sr, deep = model(lr, lr_hf)
                
                gmsd = GMSD(hr, sr)
                
                
                sr_ = quantize(sr)      
                psnr, ssim, msssim = evaluate(hr, sr_)
                        
                
                if psnr >= 40 - 2*scale:
                    soft_mask = True
                else:
                    soft_mask = False
                
                
                if soft_mask:
                    with torch.no_grad():
                        for _ in range(10): gmsd = opening(gmsd)
                    gmask = gmsd / gmsd.max()
                    gmask = (gmask > 0.2) * 1.0
                    gmask = blur(gmask)                
                    gmask = (gmask - gmask.min()) / (gmask.max() - gmask.min() + 1e-7)
                    gmask = (gmask + 0.25) / 1.25
                    gmask = gmask.detach()
                    
                    # training
                    loss = criterion(sr * gmask, hr * gmask)
                else:
                    loss = criterion(sr, hr)                
                
                # training
                loss = criterion(sr, hr)
                loss_tot = loss
                optim.zero_grad()
                loss_tot.backward()
                optim.step()
                scheduler.step()
                
                # training history 
                elapsed_time = time.time() - start_time
                elapsed = sec2time(elapsed_time)            
                pfix['Step'] = f'{step+1}'
                pfix['Loss'] = f'{loss.item():.4f}'
                
                psnrs.append(psnr)
                ssims.append(ssim)
                msssims.append(msssim)

                psnr_mean = np.array(psnrs).mean()
                ssim_mean = np.array(ssims).mean()
                msssim_mean = np.array(msssims).mean()

                pfix['PSNR'] = f'{psnr:.2f}'
                pfix['SSIM'] = f'{ssim:.4f}'
                # pfix['MSSSIM'] = f'{msssim:.4f}'
                pfix['PSNR_mean'] = f'{psnr_mean:.2f}'
                pfix['SSIM_mean'] = f'{ssim_mean:.4f}'
                # pfix['MSSSIM_mean'] = f'{msssim_mean:.4f}'
                           
                free_gpu = get_gpu_memory()[0]
                
                pfix['free GPU'] = f'{free_gpu}MiB'
                pfix['Elapsed'] = f'{elapsed}'
                
                pbar.set_postfix(pfix)
                losses.append(loss.item())
                
                if step % save_image_every == 0:
                
                    z = torch.zeros_like(lr[0])
                    _, _, llr, _ = lr.shape
                    _, _, hlr, _ = hr.shape
                    if hlr // 2 == llr:
                        xz = torch.cat((lr[0], z), dim=-2)
                    elif hlr // 4 == llr:
                        xz = torch.cat((lr[0], z, z, z), dim=-2)
                    imsave([xz, sr[0], hr[0], gmsd[0]], f'{result_dir}/epoch_{epoch+1}_iter_{step:05d}.jpg')
                    
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
                        for lr, lr_hf, hr, fname in pbar_test:
                            
                            fname = fname[0].split('/')[-1].split('.pt')[0]
                            
                            lr = lr.to(device)
                            hr = hr.to(device)
                            lr_hf = lr_hf.to(device)
                            
                            sr, deep = model(lr, lr_hf)
                            
                            mshf_hr = mshf(hr)
                            mshf_sr = mshf(sr)
                            
                            gmsd = GMSD(hr, sr)  
                            
                            sr = quantize(sr)

                            psnr, ssim, msssim = evaluate(hr, sr)

                            psnrs.append(psnr)
                            ssims.append(ssim)
                            msssims.append(msssim)

                            psnr_mean = np.array(psnrs).mean()
                            ssim_mean = np.array(ssims).mean()
                            msssim_mean = np.array(msssims).mean()

                            pfix_test['psnr'] = f'{psnr:.4f}'
                            pfix_test['ssim'] = f'{ssim:.4f}'
                            pfix_test['msssim'] = f'{msssim:.4f}'
                            pfix_test['psnr_mean'] = f'{psnr_mean:.4f}'
                            pfix_test['ssim_mean'] = f'{ssim_mean:.4f}'
                            pfix_test['msssim_mean'] = f'{msssim_mean:.4f}'
                            
                            pbar_test.set_postfix(pfix_test)
                            
                            
                            z = torch.zeros_like(lr[0])
                            _, _, llr, _ = lr.shape
                            _, _, hlr, _ = hr.shape
                            if hlr // 2 == llr:
                                xz = torch.cat((lr[0], z), dim=-2)
                            elif hlr // 4 == llr:
                                xz = torch.cat((lr[0], z, z, z), dim=-2)
                            imsave([xz, sr[0], hr[0], gmsd[0]], f'{result_dir}/{fname}.jpg')
                            
                            mshf_vis = torch.cat((torch.cat([mshf_sr[:,i,:,:] for i in range(mshf_sr.shape[1])], dim=-1),
                                                  torch.cat([mshf_hr[:,i,:,:] for i in range(mshf_hr.shape[1])], dim=-1)), dim=-2)
                            
                            imsave(mshf_vis, f'{result_dir}/MSHF_{fname}.jpg')
                            
                        hist['epoch'].append(epoch+1)
                        hist['psnr'].append(psnr_mean)
                        hist['ssim'].append(ssim_mean)
                        hist['ms-ssim'].append(msssim_mean)
                        
                        logger.add_scalar("PSNR/test", psnr_mean, epoch+1)
                        logger.add_scalar("SSIM/test", ssim_mean, epoch+1)
                        logger.add_scalar("MS-SSIM/test", msssim_mean, epoch+1)
                        
                        df = pd.DataFrame(hist)
                        df.to_csv(csv)
