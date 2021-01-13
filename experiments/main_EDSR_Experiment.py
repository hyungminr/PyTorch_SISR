import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from models.EDSR import EDSR
from utils.data_loader import get_loader
import torch

torch.manual_seed(0)


scale_factor = 4

model = EDSR(scale=scale_factor)

if scale_factor == 4:
    train_loader = get_loader(mode='train', batch_size=16, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
elif scale_factor == 2:
    train_loader = get_loader(mode='train', batch_size=16, augment=True)
    test_loader = get_loader(mode='test')

# import trainer_v1_pool as trainer
# trainer.train(model, train_loader, test_loader, mode='EDSR_v1_pool')

# import trainer_v2_centered_init as trainer
# trainer.train(model, train_loader, test_loader, mode='EDSR_v2_centered_kernel')


# import trainer_v3_grad_init as trainer
# trainer.train(model, train_loader, test_loader, mode='EDSR_v3_grad_kernel')


# import trainer as trainer
# from models.EDSR_feature_mlp import EDSR
# model = EDSR()
# trainer.train(model, train_loader, test_loader, mode='EDSR_v4_feature_mlp')


import trainer_v4_fft as trainer
trainer.train(model, train_loader, test_loader, mode='EDSR_x4_v4_fft')

# import trainer as trainer
# from models.EDSR_unshuffle import EDSR
# model = EDSR()
# trainer.train(model, train_loader, test_loader, mode='EDSR_v5_unshuffle')


# import trainer_v6_from_shallow as trainer
# from models.EDSR_train_from_shallow import EDSR
# model = EDSR(scale=scale_factor)
# trainer.train(model, train_loader, test_loader, mode='EDSR_x4_v6_from_shallow')


