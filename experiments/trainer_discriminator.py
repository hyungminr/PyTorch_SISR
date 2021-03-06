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

from collections import OrderedDict
from utils import imsave, sec2time, get_gpu_memory
from utils.eval import ssim as get_ssim
from utils.eval import ms_ssim as get_msssim
from utils.eval import psnr as get_psnr

from models.common import GMSD_quality
from models.common import MSHF

from utils import pass_filter

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



quantize = lambda x: x.mul(255).clamp(0, 255).round().div(255)

def train(model, model_sr, train_loader, test_loader, mode='EDSR_Baseline', save_image_every=50, save_model_every=10, test_model_every=1, epoch_start=0, num_epochs=1000, device=None, refresh=True):

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
    model_sr = model_sr.to(device)

    params = list(model.parameters())
    optim = torch.optim.Adam(params, lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1000, gamma= 0.99)
    criterion = torch.nn.L1Loss()
    GMSD = GMSD_quality().to(device)
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

    for epoch in range(epoch_start, epoch_start+num_epochs):

        if epoch == 0:
            torch.save(model.state_dict(), f'{weight_dir}/epoch_{epoch+1:04d}.pth')
            
        if epoch == 0:
            with torch.no_grad():
                with tqdm(test_loader, desc=f'{mode} || Warming Up || Test Epoch {epoch}/{num_epochs}', position=0, leave=True) as pbar_test:
                    psnrs = []
                    ssims = []
                    msssims = []
                    for lr, hr, fname in pbar_test:
                        lr = lr.to(device)
                        hr = hr.to(device)
                                                
                        sr, deep = model_sr(lr)
                        
                        fake = model(sr)
                        
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
            for lr, hr, _ in pbar:
                lr = lr.to(device)
                hr = hr.to(device)
                                
                # prediction
                sr, deep = model_sr(lr)
                
                fake = model(sr)
                loss_fake = criterion(fake, torch.zeros_like(fake, device=fake.device))
                
                real = model(hr)
                loss_real = criterion(real, torch.ones_like(real, device=real.device))
                
                # training
                loss_tot = loss_fake + loss_real
                optim.zero_grad()
                loss_tot.backward()
                optim.step()
                scheduler.step()
                
                # training history 
                elapsed_time = time.time() - start_time
                elapsed = sec2time(elapsed_time)            
                pfix['Step'] = f'{step+1}'
                pfix['Loss real'] = f'{loss_real.item():.4f}'
                pfix['Loss fake'] = f'{loss_fake.item():.4f}'
                
                free_gpu = get_gpu_memory()[0]
                
                pbar.set_postfix(pfix)
                step += 1
                
            if (epoch+1) % save_model_every == 0:
                torch.save(model.state_dict(), f'{weight_dir}/epoch_{epoch+1:04d}.pth')
                
            if (epoch+1) % test_model_every == 0:
                
                with torch.no_grad():
                    with tqdm(test_loader, desc=f'{mode} || Test Epoch {epoch+1}/{num_epochs}', position=0, leave=True) as pbar_test:
                        psnrs = []
                        ssims = []
                        msssims = []
                        for lr, hr, fname in pbar_test:
                                        
                            lr = lr.to(device)
                            hr = hr.to(device)
                                            
                            # prediction
                            sr, deep = model_sr(lr)
                            
                            fake = model(sr)
                            loss_fake = criterion(fake, torch.zeros_like(fake, device=fake.device))
                            
                            real = model(hr)
                            loss_real = criterion(real, torch.ones_like(real, device=real.device))
                            
                            # training history 
                            elapsed_time = time.time() - start_time
                            elapsed = sec2time(elapsed_time)            
                            pfix_test['Step'] = f'{step+1}'
                            pfix_test['Loss real'] = f'{loss_real.item():.4f}'
                            pfix_test['Loss fake'] = f'{loss_fake.item():.4f}'
                            
                            pbar_test.set_postfix(pfix_test)
                            
