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

"""
import trainer
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x{scale_factor}_Baseline')

import trainer_v1_pool as trainer
trainer.train(model, train_loader, test_loader, mode='EDSR_v1_pool')

import trainer_v2_centered_init as trainer
trainer.train(model, train_loader, test_loader, mode='EDSR_v2_centered_kernel')


import trainer_v3_grad_init as trainer
trainer.train(model, train_loader, test_loader, mode='EDSR_v3_grad_kernel')


import trainer as trainer
from models.EDSR_feature_mlp import EDSR
model = EDSR()
trainer.train(model, train_loader, test_loader, mode='EDSR_v4_feature_mlp')

import trainer as trainer
from models.EDSR_unshuffle import EDSR
model = EDSR()
trainer.train(model, train_loader, test_loader, mode='EDSR_v5_unshuffle')


import trainer_v4_fft as trainer
trainer.train(model, train_loader, test_loader, mode='EDSR_x4_v4_fft')


import trainer_v6_from_shallow as trainer
from models.EDSR_train_from_shallow import EDSR
model = EDSR(scale=scale_factor)
trainer.train(model, train_loader, test_loader, mode='EDSR_x4_v6_from_shallow')

import trainer_v6_from_shallow_all as trainer
from models.EDSR_train_from_shallow_all import EDSR
model = EDSR(scale=scale_factor)
trainer.train(model, train_loader, test_loader, mode='EDSR_x4_v6_from_shallow_all')

import trainer_v7_edge as trainer
from models.EDSR_output_edge import EDSR
model = EDSR(scale=scale_factor)
trainer.train(model, train_loader, test_loader, mode='EDSR_x4_v7_edge')

import trainer_v8_gmsd as trainer
trainer.train(model, train_loader, test_loader, mode='EDSR_x4_v8_gmsd')

import trainer_v9_rfb as trainer
from models.EDSR import EDSR
model = EDSR(scale=scale_factor)
trainer.train(model, train_loader, test_loader, mode='EDSR_x2_v9_RFB')

import trainer_v10_mshf as trainer
trainer.train(model, train_loader, test_loader, mode='EDSR_x2_v10_MSHF')


import trainer
from models.EDSR_opening import EDSR
model = EDSR(scale=scale_factor)
trainer.train(model, train_loader, test_loader, mode='EDSR_x2_v11_Opening')

import trainer_v11_gms as trainer
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x{scale_factor}_v11_gms')

import trainer_v12_gms_mshf as trainer
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x{scale_factor}_v12_gms_mshf')

import trainer_denoiser as trainer
from models.EDSR_x1 import EDSR
model = EDSR(scale=scale_factor)
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x1_denoiser')

from models.EDSR_x1 import EDSR
model = EDSR(scale=scale_factor)
import trainer_denoiser as trainer
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x2_denoise', epoch_start=0, num_epochs=100)
import trainer_deblurer as trainer
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x2_denoise_deblur', epoch_start=100, num_epochs=300)
import trainer as trainer
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x2_denoise_deblur_and_sr', epoch_start=300, num_epochs=600)


from models.EDSR_x1 import EDSR
model = EDSR()
import trainer_denoise_and_deblur as trainer
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x2_v2_dnd', epoch_start=0, num_epochs=1000)
import trainer as trainer
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x2_v2_dnd_sr', epoch_start=0, num_epochs=1000)


from models.EDSR_freq import EDSR_freq as EDSR
model = EDSR()
import trainer_v14_freq_domain as trainer
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x{scale_factor}_v14_freq_domain')


from models.EDSR_freq_concat import EDSR
model = EDSR(scale=scale_factor)
import trainer_v15_freq_domain_concat as trainer
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x{scale_factor}_v15_freq_domain_concat')


from models.EDSR_gmsd import EDSR
model = EDSR(scale=scale_factor)
import trainer_v8_gmsd as trainer
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x{scale_factor}_v8_GMSD')

import trainer_v11_gms as trainer
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x{scale_factor}_v11_GMS')

import trainer_v12_gms_mshf as trainer
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x{scale_factor}_v12_GMS_MSHF')

from models.EDSR_x1 import EDSR
model = EDSR(scale=scale_factor)
import trainer_denoise_and_deblur as trainer
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x{scale_factor}_dnd', epoch_start=0, num_epochs=1000)
import trainer_denoise_and_deblur_and_sr as trainer
# model.load_state_dict(torch.load('./weights/2021.01.20/EDSR_x4_dnd/epoch_1000.pth'))
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x{scale_factor}_dnd_sr', epoch_start=0, num_epochs=1000)


import trainer_v16_train_high_freq as trainer
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x{scale_factor}_v16_train_high_freq')


from models.EDSR_x1x2x4 import EDSR
model = EDSR(scale=4)
import trainer_v17_x1x2x4 as trainer
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x{scale_factor}_v17_x1x2x4')

from models.EDSR_freq_fusion import EDSR_fusion as EDSR
model = EDSR(scale=scale_factor)
model.EDSR_1.load_state_dict(torch.load(f'./weights/Benchmark/EDSR_x{scale_factor}.pth'))
model.EDSR_2.load_state_dict(torch.load(f'./weights/Benchmark/EDSR_x{scale_factor}.pth'))
model.EDSR_3.load_state_dict(torch.load(f'./weights/Benchmark/EDSR_x{scale_factor}.pth'))
import trainer_v18_freq_fusion as trainer
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x{scale_factor}_v18_freq_fusion')


import trainer_v19_gmsd_weighted_loss as trainer
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x{scale_factor}_v19_gmsd_weighted_loss')

import trainer_v20_gmsd_masked_loss as trainer
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x{scale_factor}_v20_gmsd_masked_loss')




import trainer_v21_gmsd_soft_masked_loss as trainer
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x{scale_factor}_v21_gmsd_soft_masked_loss')

import trainer_v22_gmsd_soft_masked_loss_vgg_perceptual as trainer
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x{scale_factor}_v22_gmsd_soft_masked_loss_vgg_perceptual')



from models.EDSR_x1 import EDSR
model = EDSR(scale=scale_factor)
# import trainer_deblurer as trainer
# trainer.train(model, train_loader, test_loader, mode=f'EDSR_x{scale_factor}_deblur', epoch_start=0, num_epochs=1000)
pretrained = torch.load('./weights/2021.01.22/EDSR_x2_deblur/epoch_1000.pth')
for n, p in model.named_parameters():
    if 'tail' not in n: p.data.copy_(pretrained[n])    
import trainer
# trainer.train(model, train_loader, test_loader, mode=f'EDSR_x{scale_factor}_v23_sr_with_pretrained_deblur')
for n, p in model.named_parameters():
    if 'tail' not in n: p.requires_grad = False
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x{scale_factor}_v24_sr_with_pretrained_deblur_fixed')


from models.EDSR_x1 import EDSR
model = EDSR(scale=scale_factor)
import trainer_deblur_nn_upscaled as trainer
trainer.train(model, train_loader, test_loader, scale=scale_factor, mode=f'EDSR_x{scale_factor}_v25_deblur_nn_upscaled')

from models.EDSR_x1 import EDSR
model = EDSR(scale=scale_factor)
import trainer_v26_nn_upsample as trainer
trainer.train(model, train_loader, test_loader, scale=scale_factor, mode=f'EDSR_x{scale_factor}_v26_nn_upscaled')


import trainer_v21_high_freq as trainer

from models.EDSR_high_freq import EDSR
scale_factor = 2
model = EDSR(scale=scale_factor)

if scale_factor == 4:
    train_loader = get_loader(mode='train', batch_size=1, height=768, width=768, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
elif scale_factor == 2:
    train_loader = get_loader(mode='train', batch_size=1, height=384, width=384, augment=True)
    test_loader = get_loader(mode='test')

trainer.train(model, train_loader, test_loader, mode=f'EDSR_x{scale_factor}_v21_high_freq')


import trainer_v27_sliding_gramm_loss as trainer
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x{scale_factor}_v27_sliding_gramm_loss')


import trainer_v28_sliding_gramm_loss_on_img as trainer
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x{scale_factor}_v28_sliding_gramm_loss_on_img')


from models.EDSR_intermediate_results import EDSR
model = EDSR(scale=scale_factor)
import trainer_v29_inter_results as trainer
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x{scale_factor}_v29_inter_results')

import trainer_v29_sliding_gramm_loss_on_prewitt as trainer
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x{scale_factor}_v29_sliding_gramm_loss_on_prewitt')


from models.EDSR_multi_loss import EDSR
model = EDSR(scale=scale_factor)
import trainer_v30_multi_loss as trainer
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x{scale_factor}_v30_multi_loss')


import trainer_v32_freqx3 as trainer
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x{scale_factor}_v32_freqx3')


from models.EDSR_x1 import EDSR
scale_factor = 2
train_loader = get_loader(mode='train', batch_size=16, augment=True)
test_loader = get_loader(mode='test')
model = EDSR(scale=scale_factor)
# import trainer_v33_postprocess as trainer
# trainer.train(model, train_loader, test_loader, scale=scale_factor, mode=f'EDSR_x{scale_factor}_v33_postprocess')
# import trainer_v34_postprocess as trainer
# trainer.train(model, train_loader, test_loader, scale=scale_factor, mode=f'EDSR_x{scale_factor}_v34_postprocess')
import trainer_v35_postprocess as trainer
trainer.train(model, train_loader, test_loader, scale=scale_factor, mode=f'EDSR_x{scale_factor}_v35_postprocess')
"""

from models.EDSR_xy import EDSR
model = EDSR(scale=scale_factor)
trainer.train(model, train_loader, test_loader, mode=f'EDSR_x{scale_factor}_v31_xy')
