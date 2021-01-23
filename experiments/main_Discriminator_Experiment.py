import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from models.EDSR import EDSR
from utils.data_loader import get_loader
import torch

torch.manual_seed(0)

scale_factor = 2

model = EDSR(scale=scale_factor)

if scale_factor == 4:
    train_loader = get_loader(mode='train', batch_size=16, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
elif scale_factor == 2:
    train_loader = get_loader(mode='train', batch_size=16, augment=True)
    test_loader = get_loader(mode='test')


from models.EDSR_x1 import EDSR
model = EDSR(scale=scale_factor)
model.load_state_dict(torch.load(f'./weights/Discriminator/EDSR_x{scale_factor}.pth'))

from models.Discriminator import VGG
model_disc = VGG(pretrained=True)

import trainer_discriminator as trainer

trainer.train(model_disc, model, train_loader, test_loader, mode=f'Discriminator_x{scale_factor}')
