import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch

"""
from models.hmnet import hmnet
from utils.data_loader import get_loader
import trainer_hmnet as trainer
torch.manual_seed(0)
scale_factor = 4
model = hmnet(scale=scale_factor)
model.load_state_dict(torch.load('./weights/2021.01.28/HMNET_x4_Baseline/epoch_0600.pth'))
if scale_factor == 4:
    train_loader = get_loader(mode='train', batch_size=16, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)    
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_Baseline', epoch_start=600, num_epochs=3000, save_model_every=100)


from models.hmnet_hf import hmnet
from utils.data_loader_freq_high import get_loader
import trainer_hmnet_v1_hf as trainer
torch.manual_seed(0)
scale_factor = 4
model = hmnet(scale=scale_factor)
if scale_factor == 4:
    train_loader = get_loader(mode='train', batch_size=8, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_v1_hf', epoch_start=0, num_epochs=3000, save_model_every=100)
from models.hmnet import hmnet
from utils.data_loader import get_loader
import trainer_hmnet as trainer
torch.manual_seed(0)
scale_factor = 4
model = hmnet(scale=scale_factor)
if scale_factor == 4:
    train_loader = get_loader(mode='train', batch_size=1, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)    
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_Baseline_batch_1', epoch_start=0, num_epochs=3000, save_model_every=100)

from models.hmnet import hmnet
from utils.data_loader import get_loader
import trainer_hmnet as trainer
torch.manual_seed(0)
scale_factor = 4
model = hmnet(scale=scale_factor)
if scale_factor == 4:
    train_loader = get_loader(mode='train', batch_size=8, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)    
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_Baseline_batch_8', epoch_start=0, num_epochs=3000, save_model_every=100)

from models.hmnet_heavy import hmnet
from utils.data_loader import get_loader
import trainer_hmnet as trainer
torch.manual_seed(0)
scale_factor = 4
model = hmnet(scale=scale_factor)
if scale_factor == 4:
    train_loader = get_loader(mode='train', batch_size=16, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)    
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_Heavy', epoch_start=0, num_epochs=3000, save_model_every=100)


from models.hmnet_heavy import hmnet
from utils.data_loader import get_loader
import trainer_hmnet as trainer
torch.manual_seed(0)
scale_factor = 4
model = hmnet(scale=scale_factor)
if scale_factor == 4:
    train_loader = get_loader(data='REDS', mode='train', batch_size=16, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(data='REDS', mode='test', height=256, width=256, scale_factor=4)    
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_Heavy_REDS', epoch_start=0, num_epochs=3000, save_model_every=1, test_model_every=10)

"""
from models.hmnet_heavy import hmnet
from utils.data_loader import get_loader
import trainer_hmnet as trainer
torch.manual_seed(0)
scale_factor = 4
model = hmnet(scale=scale_factor)
if scale_factor == 4:
    train_loader = get_loader(mode='train', batch_size=1, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)    
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_Heavy_REDS_batch_1', epoch_start=0, num_epochs=3000, save_model_every=100, test_model_every=1)
