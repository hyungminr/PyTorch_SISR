import trainer_v1_pool as trainer
from models.EDSR import EDSR
from utils.data_loader import get_loader
import torch

torch.manual_seed(0)

model = EDSR()

train_loader = get_loader(mode='train', batch_size=16, augment=True)
test_loader = get_loader(mode='test')

trainer.train(model, train_loader, test_loader, mode='EDSR_v1_pool')
