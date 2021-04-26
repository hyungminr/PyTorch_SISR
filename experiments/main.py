import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch

import datetime

from models.hmnet_heavy_ablation_fea import hmnet
from utils.data_loader import get_loader
import trainer_0426 as trainer
torch.manual_seed(0)
scale_factor = 4

batch_size = 1
epoch_start = 0
num_epochs = 10
model = hmnet(scale=scale_factor)
today = datetime.datetime.now().strftime('%Y.%m.%d')

size = 256
num_epochs = 10
train_loader = get_loader(data='REDS', mode='train', batch_size=batch_size, height=size, width=size, scale_factor=4, augment=True)
test_loader = get_loader(data='REDS', mode='test', height=256, width=256, scale_factor=4)
trainer.train(model, train_loader, test_loader, mode=f'HMNET_REDS_ab_fea', epoch_start=0, num_epochs=num_epochs, save_model_every=1, test_model_every=1, today=today, refresh=False)


