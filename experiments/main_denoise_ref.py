import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch

import datetime

from models.hmnet_heavy_x1 import hmnet
from utils.data_loader import get_loader
import trainer_hmnet_denoiser_0210 as trainer
torch.manual_seed(0)
scale_factor = 4

batch_size = 1
epoch_start = 0
num_epochs = 200

model = hmnet(scale=scale_factor)
today = datetime.datetime.now().strftime('%Y.%m.%d')
        
train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_heavy_x1_denoise', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=100, test_model_every=1, today=today)

while num_epochs <= 200:
    batch_size *= 2
    epoch_start += 200
    if batch_size == 16: num_epochs = 3000
    train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
    trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_heavy_x1_denoise', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=100, test_model_every=1, today=today, refresh=False)
    
