import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch


from models.hmnet_hf import hmnet
from utils.data_loader_freq_high import get_loader
import trainer_hmnet_v1_hf as trainer
torch.manual_seed(0)
scale_factor = 4
model = hmnet(scale=scale_factor)
if scale_factor == 4:
    train_loader = get_loader(mode='train', batch_size=16, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_v1_hf', epoch_start=0, num_epochs=1000, save_model_every=100)


from models.hmnet import hmnet
from utils.data_loader import get_loader
import trainer_hmnet as trainer
torch.manual_seed(0)
scale_factor = 4
model = hmnet(scale=scale_factor)
if scale_factor == 4:
    train_loader = get_loader(mode='train', batch_size=16, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)    
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_Baseline', epoch_start=0, num_epochs=1000, save_model_every=100)




