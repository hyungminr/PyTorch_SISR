import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch

import datetime

"""
from models.hmnet import hmnet
from utils.data_loader import get_loader
import trainer_hmnet as trainer
torch.manual_seed(0)
scale_factor = 4
model = hmnet(scale=scale_factor)
model.load_state_dict(torch.load('./weights/2021.01.28/HMNET_x4_Baseline/epoch_0600.pth'))
if scale_factor == 4:
    train_loader = get_loader(mode='train', batch_size=16, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)    
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_Baseline', epoch_start=600, num_epochs=3000, save_model_every=100)


from models.hmnet_hf import hmnet
from utils.data_loader_freq_high import get_loader
import trainer_hmnet_v1_hf as trainer
torch.manual_seed(0)
scale_factor = 4
model = hmnet(scale=scale_factor)
if scale_factor == 4:
    train_loader = get_loader(mode='train', batch_size=8, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_v1_hf', epoch_start=0, num_epochs=3000, save_model_every=100)
from models.hmnet import hmnet
from utils.data_loader import get_loader
import trainer_hmnet as trainer
torch.manual_seed(0)
scale_factor = 4
model = hmnet(scale=scale_factor)
if scale_factor == 4:
    train_loader = get_loader(mode='train', batch_size=1, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)    
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_Baseline_batch_1', epoch_start=0, num_epochs=3000, save_model_every=100)

from models.hmnet import hmnet
from utils.data_loader import get_loader
import trainer_hmnet as trainer
torch.manual_seed(0)
scale_factor = 4
model = hmnet(scale=scale_factor)
if scale_factor == 4:
    train_loader = get_loader(mode='train', batch_size=8, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)    
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_Baseline_batch_8', epoch_start=0, num_epochs=3000, save_model_every=100)

from models.hmnet_heavy import hmnet
from utils.data_loader import get_loader
import trainer_hmnet as trainer
torch.manual_seed(0)
scale_factor = 4
model = hmnet(scale=scale_factor)
if scale_factor == 4:
    train_loader = get_loader(mode='train', batch_size=16, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)    
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_Heavy', epoch_start=0, num_epochs=3000, save_model_every=100)


from models.hmnet_heavy import hmnet
from utils.data_loader import get_loader
import trainer_hmnet as trainer
torch.manual_seed(0)
scale_factor = 4
model = hmnet(scale=scale_factor)
if scale_factor == 4:
    train_loader = get_loader(data='REDS', mode='train', batch_size=16, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(data='REDS', mode='test', height=256, width=256, scale_factor=4)    
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_Heavy_REDS', epoch_start=0, num_epochs=3000, save_model_every=1, test_model_every=10)


from models.hmnet_heavy import hmnet
from utils.data_loader import get_loader
import trainer_hmnet as trainer
torch.manual_seed(0)
scale_factor = 4
model = hmnet(scale=scale_factor)
if scale_factor == 4:
    train_loader = get_loader(mode='train', batch_size=1, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)    
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_Heavy_batch_1', epoch_start=0, num_epochs=3000, save_model_every=100, test_model_every=1)


from models.hmnet_heavy import hmnet
from utils.data_loader import get_loader
import trainer_hmnet_dual_loss as trainer
torch.manual_seed(0)
scale_factor = 4
model = hmnet(scale=scale_factor)
if scale_factor == 4:
    train_loader = get_loader(mode='train', batch_size=16, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)    
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_Heavy_dual_loss', epoch_start=0, num_epochs=3000, save_model_every=100, test_model_every=1)



from models.high_pass_filter import high_pass_filter
from utils.data_loader_freq_high import get_loader
import trainer_hmnet_v2_high_pass_filter as trainer
torch.manual_seed(0)
scale_factor = 4
model = high_pass_filter(scale=scale_factor)
if scale_factor == 4:
    train_loader = get_loader(mode='train', batch_size=4, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)    
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_High_Pass_Filter', epoch_start=0, num_epochs=3000, save_model_every=100, test_model_every=1)


from models.high_pass_filter_lite import high_pass_filter
from utils.data_loader_freq_high import get_loader
import trainer_hmnet_v2_high_pass_filter as trainer
torch.manual_seed(0)
scale_factor = 4
model = high_pass_filter(scale=scale_factor)
if scale_factor == 4:
    train_loader = get_loader(mode='train', batch_size=1, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)    
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_High_Pass_Filter_lite', epoch_start=0, num_epochs=3000, save_model_every=10, test_model_every=1)


from models.hmnet_heavy import hmnet
from utils.data_loader import get_loader
import trainer_hmnet as trainer
torch.manual_seed(0)
scale_factor = 4
model = hmnet(scale=scale_factor)
batch_size = 1
epoch_start = 0
train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
# trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_Heavy_batch_1', epoch_start=epoch_start, num_epochs=200, save_model_every=100, test_model_every=1)
model.load_state_dict(torch.load('./weights/2021.02.01/HMNET_x4_Heavy_batch_1/epoch_0500.pth'))

epoch_start = 300

for _ in range(10):
    batch_size *= 2
    epoch_start += 200
    if batch_size == 32: num_epochs = 2000
    train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
    trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_Heavy_batch_{batch_size}', epoch_start=epoch_start, num_epochs=200, save_model_every=100, test_model_every=1)
    

from models.hmnet_v2 import hmnet
from utils.data_loader import get_loader
import trainer_hmnet as trainer
torch.manual_seed(0)
scale_factor = 4
model = hmnet(scale=scale_factor)
batch_size = 1
epoch_start = 0
train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_v2_batch_1', epoch_start=epoch_start, num_epochs=200, save_model_every=100, test_model_every=1)

for _ in range(10):
    batch_size *= 2
    epoch_start += 200
    train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
    trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_v2_batch_{batch_size}', epoch_start=epoch_start, num_epochs=200, save_model_every=100, test_model_every=1)
    

    
from models.hmnet_v2 import hmnet
from utils.data_loader import get_loader
import trainer_hmnet_pool_loss as trainer
torch.manual_seed(0)
scale_factor = 4
model = hmnet(scale=scale_factor)
batch_size = 1
epoch_start = 0
train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_v2_pool_loss_batch_1', epoch_start=epoch_start, num_epochs=200, save_model_every=100, test_model_every=1)

for _ in range(10):
    batch_size *= 2
    epoch_start += 200
    train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
    trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_v2_batch_{batch_size}', epoch_start=epoch_start, num_epochs=200, save_model_every=100, test_model_every=1)
    
    

from models.hmnet_v3 import hmnet
from utils.data_loader import get_loader
import trainer_hmnet as trainer
torch.manual_seed(0)
scale_factor = 4
model = hmnet(scale=scale_factor)
batch_size = 1
epoch_start = 0
num_epochs = 200
train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_v3_batch_1', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=100, test_model_every=1)
for _ in range(10):
    batch_size *= 2
    epoch_start += 200
    if batch_size == 32: num_epochs = 2000
    train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
    trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_v3_batch_{batch_size}', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=100, test_model_every=1)
    
    
from models.hmnet_v3 import hmnet
from utils.data_loader import get_loader
import trainer_hmnet_hf_loss as trainer
torch.manual_seed(0)
scale_factor = 4
model = hmnet(scale=scale_factor)
batch_size = 1
epoch_start = 0
num_epochs = 200
train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_v3_hf_loss_batch_1', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=100, test_model_every=1)
for _ in range(10):
    batch_size *= 2
    epoch_start += 200
    if batch_size == 32: num_epochs = 2000
    train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
    trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_v3_hf_loss_batch_{batch_size}', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=100, test_model_every=1)
    
  

from models.high_pass_filter_lite_v2 import high_pass_filter
from utils.data_loader_freq_high import get_loader
import trainer_hmnet_v2_high_pass_filter as trainer
torch.manual_seed(0)
scale_factor = 4
model = high_pass_filter(scale=scale_factor)
if scale_factor == 4:
    train_loader = get_loader(mode='train', batch_size=1, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)    
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_High_Pass_Filter_lite_v2', epoch_start=0, num_epochs=3000, save_model_every=10, test_model_every=1)


from models.hmnet_heavy import hmnet
from utils.data_loader import get_loader
import trainer_hmnet as trainer
torch.manual_seed(0)
scale_factor = 4
model = hmnet(scale=scale_factor)
model.load_state_dict(torch.load('./weights/2021.02.02/HMNET_x4_Heavy_batch_16/epoch_1300.pth'))
batch_size = 16
epoch_start = 1300
num_epochs = 2000
train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_Heavy_batch_{batch_size}', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=100, test_model_every=1)



from models.hmnet_v3 import hmnet
from utils.data_loader import get_loader
import trainer_hmnet_hf_loss_big as trainer
torch.manual_seed(0)
scale_factor = 4
model = hmnet(scale=scale_factor)
batch_size = 1
epoch_start = 0
num_epochs = 200
train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_v3_hf_loss_big_batch_{batch_size}', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=100, test_model_every=1)

while num_epochs == 200:
    batch_size *= 2
    epoch_start += 200
    if batch_size == 32: num_epochs = 2000
    train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
    trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_v3_hf_loss_big_batch_{batch_size}', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=100, test_model_every=1)
    
    
from models.hmnet_v3_residual import hmnet
from utils.data_loader import get_loader
import trainer_hmnet_hf_loss_big as trainer
torch.manual_seed(0)
scale_factor = 4
model = hmnet(scale=scale_factor)
batch_size = 1
epoch_start = 0
num_epochs = 200
train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_v3_residual_hf_loss_big_batch_{batch_size}', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=100, test_model_every=1)

while num_epochs == 200:
    batch_size *= 2
    epoch_start += 200
    if batch_size == 32: num_epochs = 2000
    train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
    trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_v3_residual_hf_loss_big_batch_{batch_size}', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=100, test_model_every=1)
    
    
from models.hmnet_v5 import hmnet
from utils.data_loader import get_loader
import trainer_hmnet_hf_loss_big as trainer
torch.manual_seed(0)
scale_factor = 4
model = hmnet(scale=scale_factor)
batch_size = 1
epoch_start = 0
num_epochs = 200
train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_v5_batch_{batch_size}', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=100, test_model_every=1)

while num_epochs == 200:
    batch_size *= 2
    epoch_start += 200
    if batch_size == 32: num_epochs = 2000
    train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
    trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_v5_batch_{batch_size}', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=100, test_model_every=1)
    
    
    
from models.hmnet_v4_residual_1dcnn import hmnet
from utils.data_loader import get_loader
import trainer_hmnet_hf_loss_big as trainer
torch.manual_seed(0)
scale_factor = 4
model = hmnet(scale=scale_factor)
batch_size = 1
epoch_start = 0
num_epochs = 200
train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_v4_residual_1dcnn_hf_loss_big_batch_{batch_size}', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=100, test_model_every=1)

while num_epochs == 200:
    batch_size *= 2
    epoch_start += 200
    if batch_size == 32: num_epochs = 3000
    train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
    trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_v4_residual_1dcnn_hf_loss_big_batch_{batch_size}', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=100, test_model_every=1)
    
    
    
    
from models.hmnet_v3 import hmnet
from models.post_processor import postprocessor
from utils.data_loader import get_loader
import trainer_hmnet_postprocessor as trainer
torch.manual_seed(0)
scale_factor = 4
model = hmnet(scale=scale_factor)
postmodel = postprocessor()
batch_size = 1
epoch_start = 0
num_epochs = 200

model.load_state_dict(torch.load('./weights/2021.02.04/HMNET_x4_v3_hf_loss_big_batch_32/epoch_1700.pth'))

train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
trainer.train(model, postmodel, train_loader, test_loader, mode=f'postprocessor_v1_batch_{batch_size}', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=100, test_model_every=1)

while num_epochs == 200:
    batch_size *= 2
    epoch_start += 200
    if batch_size == 32: num_epochs = 3000
    train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
    trainer.train(model, postmodel, train_loader, test_loader, mode=f'postprocessor_v1_batch_{batch_size}', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=100, test_model_every=1)
   
    
    
from models.hmnet import hmnet
from models.post_processor import postprocessor
from models.srgan_discriminator import Discriminator
from utils.data_loader import get_loader
import trainer_hmnet_gan as trainer
torch.manual_seed(0)
scale_factor = 4
model = hmnet(scale=scale_factor)
disc = Discriminator()
batch_size = 1
epoch_start = 0
num_epochs = 200
today = datetime.datetime.now().strftime('%Y.%m.%d')

train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
trainer.train(model, disc, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_GAN', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=100, test_model_every=1, today=today)

while num_epochs == 200:
    batch_size *= 2
    epoch_start += 200
    if batch_size == 32: num_epochs = 3000
    train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
    trainer.train(model, disc, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_GAN', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=100, test_model_every=1, refresh=False, today=today)
    

   
    
from models.hmnet_heavy import hmnet
from utils.data_loader import get_loader
import trainer_hmnet as trainer
torch.manual_seed(0)
scale_factor = 4

batch_size = 1
epoch_start = 0
num_epochs = 10

model = hmnet(scale=scale_factor)
today = datetime.datetime.now().strftime('%Y.%m.%d')

#train_loader = get_loader(data='REDS', mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
#test_loader = get_loader(data='REDS', mode='test', height=256, width=256, scale_factor=4)
#trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_Heavy_REDS_batch_{batch_size}', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=2, test_model_every=1, today=today)

today = '2021.02.07'
model.load_state_dict(torch.load('./weights/2021.02.07/HMNET_x4_Heavy_REDS_batch_1/epoch_0010.pth'))
while num_epochs <= 200:
    batch_size *= 2
    epoch_start += 10
    if batch_size == 32: num_epochs = 3000
    train_loader = get_loader(data='REDS', mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(data='REDS', mode='test', height=256, width=256, scale_factor=4)
    trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_Heavy_REDS_batch_{batch_size}', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=2, test_model_every=1, today=today, refresh=False)



from models.hmnet_v3 import hmnet
from utils.data_loader import get_loader
import trainer_hmnet as trainer
torch.manual_seed(0)
scale_factor = 4

batch_size = 1
epoch_start = 0
num_epochs = 200

model = hmnet(scale=scale_factor)
model.load_state_dict(torch.load('./weights/hmnet_v3_SRGAN.pth'))
today = datetime.datetime.now().strftime('%Y.%m.%d')

for n, p in model.named_parameters():
    if 'tail' not in n:
        p.requires_grad = False
        
train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_v3_from_GAN_train_tail_only', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=100, test_model_every=1, today=today)


while num_epochs <= 200:
    batch_size *= 2
    epoch_start += 200
    if batch_size == 32: num_epochs = 3000
    train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
    trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_v3_from_GAN_train_tail_only', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=100, test_model_every=1, today=today, refresh=False)


from models.hmnet_heavy_x1 import hmnet
from utils.data_loader import get_loader
import trainer_hmnet_REDS_jpeg as trainer
torch.manual_seed(0)
scale_factor = 4

batch_size = 1
epoch_start = 0
num_epochs = 200

model = hmnet(scale=scale_factor)
today = datetime.datetime.now().strftime('%Y.%m.%d')
        
train_loader = get_loader(data='REDS_jpeg', mode='train', batch_size=batch_size, height=192, width=192, scale_factor=1, augment=True)
test_loader = get_loader(data='REDS_jpeg', mode='test', height=256, width=256, scale_factor=1)
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_heavy_x1_REDS_JPEG', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=10, test_model_every=1, today=today)

while num_epochs <= 200:
    batch_size *= 2
    epoch_start += 200
    if batch_size == 16: num_epochs = 3000
    train_loader = get_loader(data='REDS_jpeg', mode='train', batch_size=batch_size, height=192, width=192, scale_factor=1, augment=True)
    test_loader = get_loader(data='REDS_jpeg', mode='test', height=256, width=256, scale_factor=1)
    trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_heavy_x1_REDS_JPEG', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=10, test_model_every=1, today=today, refresh=False)


from models.hmnet_v3 import hmnet
from utils.data_loader import get_loader
import trainer_hmnet_denoiser as trainer
torch.manual_seed(0)
scale_factor = 4

batch_size = 1
epoch_start = 0
num_epochs = 200

model = hmnet(scale=scale_factor)
today = datetime.datetime.now().strftime('%Y.%m.%d')

train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_v3_denoiser', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=100, test_model_every=1, today=today)


while num_epochs <= 200:
    batch_size *= 2
    epoch_start += 200
    if batch_size == 32: num_epochs = 3000
    train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
    trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_v3_denoiser', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=100, test_model_every=1, today=today, refresh=False)



from models.hmnet_heavy_x1 import hmnet
from utils.data_loader import get_loader
import trainer_hmnet_denoiser_0210 as trainer
torch.manual_seed(0)
scale_factor = 4

batch_size = 1
epoch_start = 0
num_epochs = 200

model = hmnet(scale=scale_factor)
today = datetime.datetime.now().strftime('%Y.%m.%d')
        
train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_heavy_x1_denoise', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=100, test_model_every=1, today=today)

while num_epochs <= 200:
    batch_size *= 2
    epoch_start += 200
    if batch_size == 16: num_epochs = 3000
    train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
    trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_heavy_x1_denoise', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=100, test_model_every=1, today=today, refresh=False)
    



from models.hmnet_heavy import hmnet
from utils.data_loader import get_loader
import trainer_hmnet as trainer
torch.manual_seed(0)
scale_factor = 4

batch_size = 1
epoch_start = 0
num_epochs = 10

model = hmnet(scale=scale_factor)
today = datetime.datetime.now().strftime('%Y.%m.%d')
today = '2021.02.16'
model.load_state_dict(torch.load('./weights/2021.02.07/HMNET_x4_Heavy_REDS_batch_32/epoch_0166.pth'))

train_loader = get_loader(data='REDS', mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
test_loader = get_loader(data='REDS', mode='test', height=256, width=256, scale_factor=4)
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_Heavy_REDS_batch_{batch_size}', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=1, test_model_every=10, today=today)

while num_epochs <= 200:
    batch_size *= 2
    epoch_start += 10
    if batch_size == 16: num_epochs = 3000
    train_loader = get_loader(data='REDS', mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(data='REDS', mode='test', height=256, width=256, scale_factor=4)
    trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_Heavy_REDS_batch_{batch_size}', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=1, test_model_every=10, today=today, refresh=False)

batch_size = 32
num_epochs = 3000
train_loader = get_loader(data='REDS', mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
test_loader = get_loader(data='REDS', mode='test', height=256, width=256, scale_factor=4)
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_Heavy_REDS_batch_{batch_size}', epoch_start=166, num_epochs=num_epochs, save_model_every=1, test_model_every=1, today=today, refresh=False)


from models.hmnet_heavy import hmnet
from utils.data_loader import get_loader
import trainer_hmnet as trainer
torch.manual_seed(0)
scale_factor = 4

batch_size = 1
epoch_start = 0
num_epochs = 10
model = hmnet(scale=scale_factor)
today = datetime.datetime.now().strftime('%Y.%m.%d')
today = '2021.02.17'
model.load_state_dict(torch.load('./weights/2021.02.07/HMNET_x4_Heavy_REDS_batch_32/epoch_0166.pth'))

size = 192 - 16
while num_epochs <= 200:
    epoch_start += 10
    size += 16
    if size > 512: num_epochs = 3000
    train_loader = get_loader(data='REDS', mode='train', batch_size=batch_size, height=size, width=size, scale_factor=4, augment=True)
    test_loader = get_loader(data='REDS', mode='test', height=256, width=256, scale_factor=4)
    trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_Heavy_REDS_size_{size}', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=1, test_model_every=2, today=today, refresh=False)



from models.hmnet_heavy import hmnet
from utils.data_loader import get_loader
import trainer_hmnet as trainer
torch.manual_seed(0)
scale_factor = 4

batch_size = 1
epoch_start = 0
num_epochs = 10
model = hmnet(scale=scale_factor)
today = datetime.datetime.now().strftime('%Y.%m.%d')
today = '2021.02.18'
model.load_state_dict(torch.load('./weights/2021.02.07/HMNET_x4_Heavy_REDS_batch_32/epoch_0166.pth'))

size = 0
num_epochs = 3000
train_loader = get_loader(data='REDS', mode='train', batch_size=batch_size, height=size, width=size, scale_factor=4, augment=True)
test_loader = get_loader(data='REDS', mode='test', height=256, width=256, scale_factor=4)
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_Heavy_REDS_size_{size}', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=1, test_model_every=1, today=today, refresh=False)
"""


from models.hmnet_heavy_x1_ablation_edge import hmnet
from utils.data_loader import get_loader
import trainer_hmnet_denoiser_0210 as trainer
torch.manual_seed(0)
scale_factor = 4

batch_size = 1
epoch_start = 0
num_epochs = 200

model = hmnet(scale=scale_factor)
today = datetime.datetime.now().strftime('%Y.%m.%d')
        
train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_heavy_x1_denoise', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=100, test_model_every=1, today=today)

while num_epochs <= 200:
    batch_size *= 2
    epoch_start += 200
    if batch_size == 16: num_epochs = 3000
    train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
    trainer.train(model, train_loader, test_loader, mode=f'HMNET_x{scale_factor}_heavy_x1_denoise_ablation_edge', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=100, test_model_every=1, today=today, refresh=False)
    

