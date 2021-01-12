<<<<<<< HEAD
import trainer_DRN as trainer
=======
<<<<<<< HEAD
import trainer_DRN
=======
import trainer_DRN as trainer
>>>>>>> b76c98a3918208af5db150361889dd3791c727f8
>>>>>>> e6994fa56381720ca2a9602b8f226e8edfaad487
from models.DRN import DRN 
from utils.data_loader import get_loader
import torch

torch.manual_seed(0)

<<<<<<< HEAD
=======
<<<<<<< HEAD
model = RCAN()

train_loader = get_loader(mode='train', batch_size=16)
test_loader = get_loader(mode='test')
=======
>>>>>>> e6994fa56381720ca2a9602b8f226e8edfaad487
model = DRN()

# train_loader = get_loader(mode='train', height=192, width=192, scale_factor=4, batch_size=4)
train_loader = get_loader(mode='train', height=96, width=96, scale_factor=4, batch_size=4)
test_loader = get_loader(mode='test', scale_factor=4)
<<<<<<< HEAD
=======
>>>>>>> b76c98a3918208af5db150361889dd3791c727f8
>>>>>>> e6994fa56381720ca2a9602b8f226e8edfaad487

trainer.train(model, train_loader, test_loader, mode='DRN_Baseline')
