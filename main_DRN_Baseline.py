import trainer_DRN as trainer
from models.DRN import DRN 
from utils.data_loader import get_loader
import torch

torch.manual_seed(0)

model = DRN()

# train_loader = get_loader(mode='train', height=192, width=192, scale_factor=4, batch_size=4)
train_loader = get_loader(mode='train', height=96, width=96, scale_factor=4, batch_size=4)
test_loader = get_loader(mode='test', scale_factor=4)

trainer.train(model, train_loader, test_loader, mode='DRN_Baseline')
