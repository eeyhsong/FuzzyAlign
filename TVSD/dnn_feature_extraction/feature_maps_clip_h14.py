"""
Obtain CLIP features of training and test images in Things-EEG1.

using huggingface pretrained CLIP model

"""

import argparse
import torch.nn as nn
import numpy as np
import torch
import os
from PIL import Image
import pandas as pd
import open_clip

gpus = [1]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
os.environ['HF_ENDPOINT']= 'https://hf-mirror.com'


# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--dnn', default='clip_h14', type=str)
parser.add_argument('--project_dir', default='./data/THINGS/', type=str)
parser.add_argument('--data_dir', default='./data/TVSD/', type=str)
parser.add_argument('--train_csv', default='./TVSD/things_imgs_train.csv', type=str)
parser.add_argument('--test_csv', default='./TVSD/things_imgs_test.csv', type=str)
args = parser.parse_args()

print('Extract feature maps CLIP <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

seed = 20200220
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k', precision='fp32') 
model = model.cuda()
model = nn.DataParallel(model, device_ids=[i for i in range(len(gpus))])

condi_path = pd.read_csv(args.train_csv)
condi_path = condi_path.iloc[:, 0]

# save into list
condi_path_test = pd.read_csv(args.test_csv)
condi_path_test = condi_path_test.iloc[:, 0]
train_feats = []
test_feats = []

save_dir = os.path.join(args.data_dir, 'DNN_feature_maps', args.dnn)
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

# training feature maps
# list images of each condition
for condi_idx in range(len(condi_path)):
	condi_path[condi_idx] = condi_path[condi_idx].replace('\\', '/')
	img_path = os.path.join(args.project_dir, 'Images', condi_path[condi_idx].replace('\\', '/')[1:-1])

	image = preprocess(Image.open(img_path)).unsqueeze(0).cuda()
	x = model.module.visual(image)
	feats = x.detach().cpu().numpy()
	train_feats.append(feats)
	

# save feature maps
train_feats = np.array(train_feats)
np.save(os.path.join(save_dir, args.dnn+'_feats_train.npy'), train_feats)

print('111')

for condi_idx in range(100):
	one_cond_dir = os.path.join(args.project_dir, 'Images', condi_path_test[condi_idx][1:-1].split('\\')[0])
	cond_img_list = os.listdir(one_cond_dir)
	cond_img_list.sort() 
	cond_center =[]
	for img in cond_img_list:
		if img != condi_path_test[condi_idx][1:-1].split('\\')[1]:
			continue
		img_path = os.path.join(one_cond_dir, img)
		image = preprocess(Image.open(img_path)).unsqueeze(0).cuda()
		with torch.no_grad():
			outputs = model.module.visual(image)
	
			feats = outputs.detach().cpu().numpy()
			test_feats.append(feats)

test_feats = np.array(test_feats)
np.save(os.path.join(save_dir, args.dnn+'_feats_test.npy'), test_feats)


print('111')


# center feats
all_centers = []

for condi_idx in range(100):
	one_cond_dir = os.path.join(args.project_dir, 'Images', condi_path_test[condi_idx][1:-1].split('\\')[0])
	cond_img_list = os.listdir(one_cond_dir)
	cond_img_list = [f for f in cond_img_list if not f.startswith('._')]
	cond_img_list.sort() 
	cond_center =[]
	for img in cond_img_list:
		if img == condi_path_test[condi_idx][1:-1].split('\\')[1]:
			continue
		img_path = os.path.join(one_cond_dir, img)
		image = preprocess(Image.open(img_path)).unsqueeze(0).cuda()
		with torch.no_grad():
			outputs = model.module.visual(image)
		cond_center.append(outputs.detach().cpu().numpy())
	cond_center = np.mean(cond_center, axis=0)
	all_centers.append(np.squeeze(cond_center))

np.save(os.path.join(save_dir, 'center_' + args.dnn+'.npy'), np.array(all_centers))

print('111')
