import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils.data_loader import get_loader
import torch

torch.manual_seed(0)

from models.RCAN import RCAN
model = RCAN()

scale_factor = 2

if scale_factor == 4:
    train_loader = get_loader(mode='train', batch_size=16, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
elif scale_factor == 2:
    train_loader = get_loader(mode='train', batch_size=16, augment=True)
    test_loader = get_loader(mode='test')

# import trainer as trainer
# trainer.train(model, train_loader, test_loader, mode='RCAN_x2_Baseline')
    
# import trainer_v6_from_shallow as trainer
# from models.RCAN_train_from_shallow import RCAN
# model = RCAN()
# trainer.train(model, train_loader, test_loader, mode='RCAN_v6_from_shallow')

import trainer_v8_gmsd as trainer
trainer.train(model, train_loader, test_loader, mode='RCAN_x2_v8_gmsd_pretrained')

# import trainer_v10_mshf as trainer
# trainer.train(model, train_loader, test_loader, mode='RCAN_x2_v10_MSHF')
