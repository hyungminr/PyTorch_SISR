import torch

import torchvision.transforms as T

import glob
from PIL import Image
from tqdm import tqdm
import os
from utils import high_pass_filter_hard_kernel

transform = []
transform.append(T.ToTensor())
t = T.Compose(transform)



data_dir = []
# data_dir.append('./data/DIV2K/DIV2K_train_HR/*png')
# data_dir.append('./data/DIV2K/DIV2K_valid_HR/*png')
data_dir.append('./data/DIV2K/DIV2K_train_LR_bicubic/X2/*png')
data_dir.append('./data/DIV2K/DIV2K_valid_LR_bicubic/X2/*png')
data_dir.append('./data/DIV2K/DIV2K_train_LR_bicubic/X4/*png')
data_dir.append('./data/DIV2K/DIV2K_valid_LR_bicubic/X4/*png')

for d in data_dir:
    images = glob.glob(d)
    for iname in tqdm(images):
        img = Image.open(iname)
        tensor = t(img)
        tensor = high_pass_filter_hard_kernel(tensor.unsqueeze(0))
        tensor = tensor.squeeze(0)
        rname = iname.replace('/DIV2K/', '/DIV2K/bin/hfreq/')
        rname = rname.replace('.png', '.pt')
        os.makedirs('/'.join(rname.split('/')[:-1]), exist_ok=True)
        torch.save(tensor, rname)
