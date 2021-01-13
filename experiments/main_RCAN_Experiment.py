import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils.data_loader import get_loader
import torch

torch.manual_seed(0)

train_loader = get_loader(mode='train', batch_size=16, augment=True)
test_loader = get_loader(mode='test')

import trainer_v6_from_shallow as trainer
from models.RCAN_train_from_shallow import RCAN
model = RCAN()
trainer.train(model, train_loader, test_loader, mode='RCAN_v6_from_shallow')


