import torch
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

from diffusers import AutoencoderKL

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default='preset/datasets/train_datasets/training', help='the dataset you want to tag.') # 
parser.add_argument("--save_path", type=str, default='preset/datasets/train_datasets/training', help='the dataset you want to tag.') # 
parser.add_argument("--end_num", type=int, default=-1)
parser.add_argument("--start_num", type=int, default=0)
args = parser.parse_args()

def PrintInfo(x):
    if not isinstance(x,list):
        x=[x]
    for i in x:
        print('shape : {} ; dtype : {} ; max : {} ; min : {}'.format(i.shape,i.dtype,i.max(),i.min())  )

img_preproc = transforms.Compose([       
            transforms.ToTensor(),
        ])
img_afterproc = transforms.Compose([ 
            transforms.ToPILImage(),
        ])

img_folder = args.root_path
img_save_folder = args.save_path
os.makedirs(img_save_folder, exist_ok=True)
img_name_list = os.listdir(img_folder)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = AutoencoderKL.from_pretrained('preset/models/stable-diffusion-3.5-medium', subfolder="vae", revision=None)
vae.requires_grad_(False)
vae.to(device)

for img_name in tqdm(img_name_list[args.start_num:args.end_num]):
    if os.path.exists(os.path.join(img_save_folder, img_name.replace('png', 'pt'))):
        continue
    img_path = os.path.join(img_folder, img_name)
    img = Image.open(img_path).convert('RGB')
    img = img_preproc(img)
    img = img * 2.0 - 1.0
    img = img.unsqueeze(0)
    img = img.to(device)
    with torch.no_grad():
        latents = vae.encode(img).latent_dist.sample()
        latents = (latents - vae.config.shift_factor)  * vae.config.scaling_factor
    torch.save(latents.clone(), os.path.join(img_save_folder, img_name.replace('png', 'pt')))