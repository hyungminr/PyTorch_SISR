
from models.EDSR import EDSR
from utils.data_loader import get_loader
import torch

torch.manual_seed(0)

model = EDSR()

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


# import trainer_v4_fft as trainer
# trainer.train(model, train_loader, test_loader, mode='EDSR_v4_fft')

import trainer_v5_unshuffle as trainer
from models.EDSR_unshuffle import EDSR
model = EDSR()
trainer.train(model, train_loader, test_loader, mode='EDSR_v5_unshuffle')
