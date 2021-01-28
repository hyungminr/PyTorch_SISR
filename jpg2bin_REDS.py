import torch

import torchvision.transforms as T

import glob
from PIL import Image
from tqdm import tqdm
import os

transform = []
transform.append(T.ToTensor())
t = T.Compose(transform)

data_dir = []
data_dir.append('./data/benchmark/REDS/train/train_sharp/*/*.png')
data_dir.append('./data/benchmark/REDS/train/train_blur_bicubic/X4/*/*.png')
data_dir.append('./data/benchmark/REDS/val/val_sharp/*/*.png')
data_dir.append('./data/benchmark/REDS/val/val_blur_bicubic/X4/*/*.png')
data_dir.append('./data/benchmark/REDS/test/test_blur_bicubic/X4/*/*.png')

for d in data_dir:
    print(d)
    images = glob.glob(d)
    for iname in tqdm(images):
        img = Image.open(iname)
        tensor = t(img)
        rname = iname.replace('/REDS/', '/REDS/bin/')
        rname = rname.replace('.png', '.pt')
        os.makedirs('/'.join(rname.split('/')[:-1]), exist_ok=True)
        torch.save(tensor, rname)


