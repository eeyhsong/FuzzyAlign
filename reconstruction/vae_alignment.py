import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import TensorDataset
import numpy as np
import random
import torch.nn as nn
import argparse
import time
from diffusers import DiffusionPipeline
import torchvision.utils as vutils
try:
    from .model import vae_latent_projector
except ImportError:
    from model import vae_latent_projector

parser = argparse.ArgumentParser(description="Visual stimuli reconstruction with EEG")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser.add_argument('--save_path', default='./generated_imgs_low_level/', type=str)
parser.add_argument('--models_root', default='./models', type=str)
parser.add_argument('--vae_latent_root', default='./data/TVSD/DNN_feature_maps/vae_latents', type=str)
parser.add_argument('--test_index_path', default='./TVSD/things_imgs_test.csv', type=str)
parser.add_argument('--test_idx_path', default='./TVSD/test_idx_intrain.npy', type=str)
parser.add_argument('--dnn', default='clip_h14', type=str)
parser.add_argument('--batch_size', default=200, type=int, metavar='N', help='batch size')
parser.add_argument('--seed', default=2024, type=int, help='seed for initializing training')
parser.add_argument('--val', default='GA_wo_baseline_corr_test', type=str)
parser.add_argument('--num_epochs', default=150, type=int)


def get_eeg_features(args, nSub):
    model_dir = os.path.join(args.models_root, args.dnn, args.val, nSub)
    train_eeg_features = torch.load(os.path.join(model_dir, 'eeg_train_features.pt'), weights_only=True)['eeg_train_features']
    test_eeg_features = torch.load(os.path.join(model_dir, 'eeg_test_features.pt'), weights_only=True)['eeg_test_features']
    
    return train_eeg_features, test_eeg_features

def get_vae_features(args):
    train_vae_latents = torch.load(os.path.join(args.vae_latent_root, 'train_vae_latents.pt'), weights_only=True)['train_vae_latents']
    test_vae_latents = torch.load(os.path.join(args.vae_latent_root, 'test_vae_latents.pt'), weights_only=True)['test_vae_latents']
    
    return train_vae_latents, test_vae_latents

def load_category(args):
    data_path = args.test_index_path
    condi_path_test = pd.read_csv(data_path)
    condi_path_test = condi_path_test.iloc[:, 0]

    texts = []
    for condi_idx in range(100):
        category_name = condi_path_test[condi_idx][1:-1].split('\\')[0]
        texts.append(category_name)
    return texts

def main(): 
    args = parser.parse_args()
    subjects = ['sub-01', 'sub-02']

    seed_n = args.seed
    print('seed is ' + str(seed_n))
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    categories = load_category(args)
    
    for sub in subjects:
        print('The subject is: ', sub)
        print('-----------------------------------------------------------------------------------------------------')
        train_eeg_features, test_eeg_features = get_eeg_features(args, sub)
        train_vae_latents, test_vae_latents = get_vae_features(args)
        test_idx_intrain = torch.from_numpy(np.load(args.test_idx_path)).long()
        if train_vae_latents.shape[0] != train_eeg_features.shape[0]:
            keep_mask = torch.ones(train_vae_latents.shape[0], dtype=torch.bool)
            keep_mask[test_idx_intrain] = False
            train_vae_latents = train_vae_latents[keep_mask]
        print('The shape of train_eeg_features is:', train_eeg_features.shape)
        print('The shape of test_eeg_features  is:', test_eeg_features.shape)
        print('The shape of train_vae_latents is:', train_vae_latents.shape)
        print('The shape of test_vae_latents is:', test_vae_latents.shape)

        assert train_eeg_features.shape[0] == train_vae_latents.shape[0], "Train EEG and VAE latent count mismatch"
        assert test_eeg_features.shape[0] == test_vae_latents.shape[0], "Test EEG and VAE latent count mismatch"
        assert torch.isfinite(train_vae_latents).all(), "train_vae_latents has NaN/Inf"
        assert torch.isfinite(test_vae_latents).all(), "test_vae_latents has NaN/Inf"

        print('-----------------------------------------------------------------------------------------------------')
        print("EEG-to-image embeddings Minimum:", train_eeg_features.min().item())
        print("EEG-to-image embeddings Maximum:", train_eeg_features.max().item())
        print('-----------------------------------------------------------------------------------------------------')
        print("Image vae latents Minimum:", train_vae_latents.min().item())
        print("Image vae latents Maximum:", train_vae_latents.max().item())
        print('-----------------------------------------------------------------------------------------------------')
        # shuffle the training data
        train_shuffle = np.random.permutation(len(train_eeg_features))
        train_eeg_features = train_eeg_features[train_shuffle]
        train_vae_latents = train_vae_latents[train_shuffle]

        val_eeg_features = train_eeg_features[:748]
        val_vae_latents = train_vae_latents[:748]

        train_eeg_features = train_eeg_features[748:]
        train_vae_latents = train_vae_latents[748:]
        
        # Prepare data loaders
        train_dataset = TensorDataset(train_eeg_features, train_vae_latents)
        val_dataset = TensorDataset(val_eeg_features, val_vae_latents)
        test_dataset = TensorDataset(test_eeg_features, test_vae_latents)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model = vae_latent_projector().to(torch.float32).to(device)
        print('The number of parameters of low level model:', sum([p.numel() for p in model.parameters()]))
        criterion_mse = nn.MSELoss().to(device)

        optimizer = optim.AdamW(model.parameters(), lr=3e-4)

        num_epochs = args.num_epochs

        # Training loop
        best_val_loss = np.inf
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(torch.float32).to(device), targets.to(torch.float32).to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion_mse(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss = train_loss / (batch_idx + 1)
            
            if (epoch + 1) % 1 == 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for idx, (inputs, targets) in enumerate(val_loader):
                        inputs, targets = inputs.to(torch.float32).to(device), targets.to(torch.float32).to(device)
                        outputs = model(inputs)
                        loss = criterion_mse(outputs, targets)
                        val_loss += loss.item()
                    val_loss = val_loss / (idx + 1)
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_dir = os.path.join(args.models_root, args.dnn, args.val, sub)
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(model_dir, 'vae_condition.pth'))
                    print("Model saved as vae_condition.pth")

        pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
        vae = pipe.vae.to(device, dtype=torch.float32)
        vae.requires_grad_(False)
        vae.eval()

        # Testing loop
        model_dir = os.path.join(args.models_root, args.dnn, args.val, sub)
        model.load_state_dict(torch.load(os.path.join(model_dir, 'vae_condition.pth'), weights_only=True), strict=False)
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(torch.float32).to(device), targets.to(torch.float32).to(device)
                outputs = model(inputs)
                loss = criterion_mse(outputs, targets)
                test_loss += loss.item()
                
                image_recon = vae.decode(outputs / vae.config.scaling_factor, return_dict=False)[0]
                image_recon = (image_recon / 2 + 0.5).clamp(0, 1).float().detach()
                save_path = os.path.join(args.save_path, f"{args.dnn}/{args.val}/{sub}/{categories[idx]}/low_level_recon.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                vutils.save_image(image_recon.cpu(), save_path)
                                   
            test_loss = test_loss / (idx + 1)
        print(f"Test Loss: {test_loss}")

        vae_model = vae_latent_projector().to(torch.float32).to(device)
        vae_model.load_state_dict(torch.load(os.path.join(model_dir, 'vae_condition.pth'), weights_only=True), strict=False)
        # Testing loop
        vae_model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(torch.float32).to(device), targets.to(torch.float32).to(device)
                outputs = vae_model(inputs)
                loss = criterion_mse(outputs, targets)
                test_loss += loss.item()
            test_loss = test_loss / (idx + 1)
        print(f"After loading, the test Loss: {test_loss}")
        
if __name__ == '__main__':
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))
