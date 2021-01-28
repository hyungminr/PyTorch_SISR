import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from models.hmnet import hmnet
from utils.data_loader import get_loader
import torch

torch.manual_seed(0)


scale_factor = 4

model = EDSR(scale=scale_factor)

if scale_factor == 4:
    train_loader = get_loader(mode='train', batch_size=16, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
    
import trainer_hmnet as trainer
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_Baseline')
