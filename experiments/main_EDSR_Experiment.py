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

# import trainer
# trainer.train(model, train_loader, test_loader, mode='EDSR_x2_Baseline')

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

# import trainer as trainer
# from models.EDSR_unshuffle import EDSR
# model = EDSR()
# trainer.train(model, train_loader, test_loader, mode='EDSR_v5_unshuffle')


# import trainer_v4_fft as trainer
# trainer.train(model, train_loader, test_loader, mode='EDSR_x4_v4_fft')


# import trainer_v6_from_shallow as trainer
# from models.EDSR_train_from_shallow import EDSR
# model = EDSR(scale=scale_factor)
# trainer.train(model, train_loader, test_loader, mode='EDSR_x4_v6_from_shallow')

# import trainer_v6_from_shallow_all as trainer
# from models.EDSR_train_from_shallow_all import EDSR
# model = EDSR(scale=scale_factor)
# trainer.train(model, train_loader, test_loader, mode='EDSR_x4_v6_from_shallow_all')

# import trainer_v7_edge as trainer
# from models.EDSR_output_edge import EDSR
# model = EDSR(scale=scale_factor)
# trainer.train(model, train_loader, test_loader, mode='EDSR_x4_v7_edge')

# import trainer_v8_gmsd as trainer
# trainer.train(model, train_loader, test_loader, mode='EDSR_x4_v8_gmsd')


# import trainer_v9_rfb as trainer
# from models.EDSR import EDSR
# model = EDSR(scale=scale_factor)
# trainer.train(model, train_loader, test_loader, mode='EDSR_x2_v9_RFB')



# import trainer_v10_mshf as trainer
# trainer.train(model, train_loader, test_loader, mode='EDSR_x2_v10_MSHF')


# import trainer
# from models.EDSR_opening import EDSR
# model = EDSR(scale=scale_factor)
# trainer.train(model, train_loader, test_loader, mode='EDSR_x2_v11_Opening')


import trainer_v11_gms as trainer
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x{scale_factor}_v11_gms')
