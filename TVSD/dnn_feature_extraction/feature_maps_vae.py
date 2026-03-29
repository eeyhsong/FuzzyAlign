import torch
import numpy as np
import os
import argparse
from torch.nn import functional as F
from PIL import Image
import random
import pandas as pd
from diffusers import DiffusionPipeline

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--dnn', default='clip_h14', type=str)
parser.add_argument('--project_dir', default='./data/THINGS/Images/', type=str)
parser.add_argument('--save_dir', default='./data/TVSD/DNN_feature_maps/vae_latents/', type=str)
parser.add_argument('--train_csv', default='./TVSD/things_imgs_train.csv', type=str)
parser.add_argument('--test_csv', default='./TVSD/things_imgs_test.csv', type=str)
args = parser.parse_args()
os.makedirs(args.save_dir, exist_ok=True)

model_type = 'SDXL'
pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
vae = pipe.vae.to(device, dtype=torch.float32)
vae.requires_grad_(False)
vae.eval()
print('The vae_scale_factor is:', vae.config.scaling_factor)


seed_value = 2024
print('The seed is ' + str(seed_value))
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)


condi_path = pd.read_csv(args.train_csv)
condi_path = condi_path.iloc[:, 0]
train_paths = []
for condi_idx in range(len(condi_path)):
    img_path = os.path.join(args.project_dir, condi_path[condi_idx].replace('\\', '/')[1:-1])
    train_paths.append(img_path)

condi_path_test = pd.read_csv(args.test_csv)
condi_path_test = condi_path_test.iloc[:, 0]
test_paths = []
for condi_idx in range(len(condi_path_test)):
    img_path = os.path.join(args.project_dir, condi_path_test[condi_idx].replace('\\', '/')[1:-1])
    test_paths.append(img_path)

train_feats_list = []
test_feats_list = []

batch_size = 20
for i in range(0, len(test_paths), batch_size):
    batch_images = test_paths[i:i + batch_size]
    image_inputs = torch.stack([pipe.image_processor.preprocess(Image.open(img).convert("RGB"), height=512, width=512) for img in batch_images]).to(device)
    image_inputs = image_inputs.squeeze(1)
    print('The shape of test_image_inputs is:', image_inputs.shape)
    with torch.no_grad():
        batch_image_features = vae.encode(image_inputs).latent_dist.mode() * vae.config.scaling_factor
        batch_image_features = batch_image_features.cpu()
        batch_image_features = F.normalize(batch_image_features, dim=-1).detach() # todo
    print('The shape of test_batch_image_features is:', batch_image_features.shape)
    print('-----------------------------------------')
    test_feats_list.append(batch_image_features)

test_feats = torch.cat(test_feats_list, dim=0)
print('The shape of test_feats is: ', test_feats.shape)

for i in range(0, len(train_paths), batch_size):
    print('The index is:', i)
    batch_images = train_paths[i:i + batch_size]
    image_inputs = torch.stack([pipe.image_processor.preprocess(Image.open(img).convert("RGB"), height=512, width=512) for img in batch_images]).to(device)
    image_inputs = image_inputs.squeeze(1)
    print('The shape of train_image_inputs is:', image_inputs.shape)
    with torch.no_grad():
        batch_image_features = vae.encode(image_inputs).latent_dist.mode() * vae.config.scaling_factor
        batch_image_features = batch_image_features.cpu()
        batch_image_features = F.normalize(batch_image_features, dim=-1).detach() # todo
    print('The shape of train_batch_image_features is:', batch_image_features.shape)
    print('-----------------------------------------')
    train_feats_list.append(batch_image_features)

# save feature maps
train_feats = torch.cat(train_feats_list, dim=0)
print('The shape of train_feats is: ', train_feats.shape)

torch.save({'train_vae_latents': train_feats}, os.path.join(args.save_dir, 'train_vae_latents.pt'))
torch.save({'test_vae_latents': test_feats}, os.path.join(args.save_dir, 'test_vae_latents.pt'))

