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
from models.common import GMSD as GMS
from models.common import MSHF

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

def train(model, train_loader, test_loader, mode='EDSR_Baseline', save_image_every=50, save_model_every=10, test_model_every=1, num_epochs=1000, device=None, refresh=True):

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

    ######
    
    GMSD_map = GMS(pad=0).to(device)
    GMSD = GMSD_quality().to(device)
    mshf = MSHF(3, 3).to(device)
    
    ######

    start_time = time.time()
    print(f'Training Start || Mode: {mode}')

    step = 0
    pfix = OrderedDict()
    pfix_test = OrderedDict()

    hist = dict()
    hist['mode'] = f'{today}_{mode}'
    for key in ['epoch', 'psnr', 'ssim', 'ms-ssim']:
        hist[key] = []

    pretrained_epoch = 200

    for epoch in range(num_epochs):
        if epoch < pretrained_epoch: continue
        if epoch == pretrained_epoch:
           model.load_state_dict(torch.load(f'./weights/2021.01.15/EDSR_x2_Baseline/epoch_{pretrained_epoch:04d}.pth'))
           # model.load_state_dict(torch.load(f'./weights/2021.01.17/RCAN_x2_Baseline/epoch_{pretrained_epoch:04d}.pth'))
        if epoch == 0:
           torch.save(model.state_dict(), f'{weight_dir}/epoch_{epoch+1:04d}.pth')
            
        if epoch == 0:
            with torch.no_grad():
                with tqdm(test_loader, desc=f'Mode: {mode} || Warming Up || Test Epoch {epoch}/{num_epochs}', position=0, leave=True) as pbar_test:
                    psnrs = []
                    ssims = []
                    msssims = []
                    for lr, hr, fname in pbar_test:
                        lr = lr.to(device)
                        hr = hr.to(device)
                        
                        sr, features = model(lr)
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
                        

        with tqdm(train_loader, desc=f'Mode: {mode} || Epoch {epoch+1}/{num_epochs}', position=0, leave=True) as pbar:
            psnrs = []
            ssims = []
            msssims = []
            losses = []
            for lr, hr, _ in pbar:
                lr = lr.to(device)
                hr = hr.to(device)
                
                # prediction
                sr, features = model(lr)
                
                ####
                gmsd_sr = GMSD_map(sr)   
                gmsd_hr = GMSD_map(hr)   
                gmsd = GMSD(hr, sr)     
                mshf_hr = mshf(hr)
                mshf_sr = mshf(sr)
                        
                loss_gmsd = criterion(gmsd_hr, gmsd_sr)
                loss_mshf = criterion(mshf_hr, mshf_sr)
                #####
                
                # training
                loss = criterion(hr, sr)
                loss_tot = loss + 0.2 * (loss_gmsd + loss_mshf)
                optim.zero_grad()
                loss_tot.backward()
                optim.step()
                scheduler.step()
                
                # training history 
                elapsed_time = time.time() - start_time
                elapsed = sec2time(elapsed_time)            
                pfix['Step'] = f'{step+1}'
                pfix['Loss'] = f'{loss.item():.4f}'
                pfix['Loss GMSD'] = f'{loss_gmsd.item():.4f}'
                
                sr = quantize(sr)      
                psnr, ssim, msssim = evaluate(hr, sr)
                        
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
                    with tqdm(test_loader, desc=f'Mode: {mode} || Test Epoch {epoch+1}/{num_epochs}', position=0, leave=True) as pbar_test:
                        psnrs = []
                        ssims = []
                        msssims = []
                        for lr, hr, fname in pbar_test:
                            
                            fname = fname[0].split('/')[-1].split('.pt')[0]
                            
                            lr = lr.to(device)
                            hr = hr.to(device)

                            sr, features = model(lr)
                            
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
                            
                        hist['epoch'].append(epoch+1)
                        hist['psnr'].append(psnr_mean)
                        hist['ssim'].append(ssim_mean)
                        hist['ms-ssim'].append(msssim_mean)
                        
                        logger.add_scalar("PSNR/test", psnr_mean, epoch+1)
                        logger.add_scalar("SSIM/test", ssim_mean, epoch+1)
                        logger.add_scalar("MS-SSIM/test", msssim_mean, epoch+1)
                        
                        df = pd.DataFrame(hist)
                        df.to_csv(csv)
