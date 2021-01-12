<<<<<<< HEAD
import trainer_DRN
=======
import trainer_DRN as trainer
>>>>>>> b76c98a3918208af5db150361889dd3791c727f8
from models.DRN import DRN 
from utils.data_loader import get_loader
import torch

torch.manual_seed(0)

<<<<<<< HEAD
model = RCAN()

train_loader = get_loader(mode='train', batch_size=16)
test_loader = get_loader(mode='test')
=======
model = DRN()

# train_loader = get_loader(mode='train', height=192, width=192, scale_factor=4, batch_size=4)
train_loader = get_loader(mode='train', height=96, width=96, scale_factor=4, batch_size=4)
test_loader = get_loader(mode='test', scale_factor=4)
>>>>>>> b76c98a3918208af5db150361889dd3791c727f8

trainer.train(model, train_loader, test_loader, mode='DRN_Baseline')
