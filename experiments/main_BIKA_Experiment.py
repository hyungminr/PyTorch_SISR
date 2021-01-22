import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from models.BIKANet import BIKA
from utils.data_loader import get_loader
import torch

torch.manual_seed(0)


scale_factor = 2

model = BIKA(scale=scale_factor)

if scale_factor == 4:
    train_loader = get_loader(mode='train', batch_size=16, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
elif scale_factor == 2:
    train_loader = get_loader(mode='train', batch_size=16, augment=True)
    test_loader = get_loader(mode='test')
"""
import trainer_bika as trainer
trainer.train(model, train_loader, test_loader, mode=f'BIKA_x{scale_factor}_Baseline')
"""
"""
import trainer_bika_down_nn as trainer
trainer.train(model, train_loader, test_loader, mode=f'BIKA_x{scale_factor}_Down_nn')
"""

import trainer_bika_pretrain_deblur as trainer
trainer.train(model, train_loader, test_loader, scale=scale_factor, mode=f'BIKA_x{scale_factor}_Deblur_Pretrain')

