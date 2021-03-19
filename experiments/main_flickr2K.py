import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch

import datetime

today='2021.03.20'


from models.hmnet_heavy_x1 import hmnet
from utils.data_loader import get_loader
import trainer_hmnet_Flickr2K as trainer
torch.manual_seed(0)
scale_factor = 4

batch_size = 1
epoch_start = 0
num_epochs = 200

model = hmnet(scale=scale_factor)
#model.load_state_dict(torch.load('./weights/HMNET_x4_Heavy_REDS_JPEG.pth'))

train_loader = get_loader(data='Flickr2K', mode='train', batch_size=batch_size, height=0, width=0, scale_factor=1, augment=True, force_size=True)
test_loader = get_loader(data='Flickr2K', mode='test', batch_size=batch_size, height=0, width=0, scale_factor=1, augment=True, force_size=True)
trainer.train(model, train_loader, test_loader, mode=f'HMNET_Flickr2K', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=1, test_model_every=1, today=today)
    
from models.hmnet_heavy_x1_ab_fea_0310 import hmnet
from utils.data_loader import get_loader
import trainer_hmnet_Flickr2K as trainer
torch.manual_seed(0)
scale_factor = 4

batch_size = 1
epoch_start = 0
num_epochs = 200

model = hmnet(scale=scale_factor)
#model.load_state_dict(torch.load('./weights/HMNET_x4_Heavy_REDS_JPEG.pth'))


train_loader = get_loader(data='Flickr2K', mode='train', batch_size=batch_size, height=0, width=0, scale_factor=1, augment=True, force_size=True)
test_loader = get_loader(data='Flickr2K', mode='test', batch_size=batch_size, height=0, width=0, scale_factor=1, augment=True, force_size=True)
trainer.train(model, train_loader, test_loader, mode=f'HMNET_Flickr2K_ablation_fea', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=1, test_model_every=1, today=today)
    
from models.hmnet_heavy_x_ab_edge_0310 import hmnet
from utils.data_loader import get_loader
import trainer_hmnet_Flickr2K as trainer
torch.manual_seed(0)
scale_factor = 4

batch_size = 1
epoch_start = 0
num_epochs = 200

model = hmnet(scale=scale_factor)
#model.load_state_dict(torch.load('./weights/HMNET_x4_Heavy_REDS_JPEG.pth'))

train_loader = get_loader(data='Flickr2K', mode='train', batch_size=batch_size, height=0, width=0, scale_factor=1, augment=True, force_size=True)
test_loader = get_loader(data='Flickr2K', mode='test', batch_size=batch_size, height=0, width=0, scale_factor=1, augment=True, force_size=True)
trainer.train(model, train_loader, test_loader, mode=f'HMNET_Flickr2K_ablation_edge', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=1, test_model_every=1, today=today)
    
