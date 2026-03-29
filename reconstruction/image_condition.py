import os
import torch
from torch.nn import functional as F
import time
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch import Tensor
import torch.nn as nn
try:
    from .model import BrainDiffusionPrior, PriorNetwork
except ImportError:
    from model import BrainDiffusionPrior, PriorNetwork
from einops.layers.torch import Rearrange
import random
import argparse
from einops import rearrange
import math
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Visual stimuli reconstruction with EEG")
parser.add_argument('--eeg_data_path', default='./data/TVSD/Preprocessed_data/', type=str)
parser.add_argument('--img_data_path', default='./data/TVSD/DNN_feature_maps/', type=str)
parser.add_argument('--models_root', default='./models', type=str)
parser.add_argument('--dnn', default='clip_h14', type=str)

parser.add_argument('--lr', default=3e-4, type=float, help='learning rate for model training')
parser.add_argument('--in_dim', default=1024, type=int, help='the dimension of input')
parser.add_argument('--num_tokens', default=1, type=int, help='the number of text tokens')
parser.add_argument('--clip_dim', default=1024, type=int, help='the dimension of clip text embeddings')
parser.add_argument('--n_blocks', default=2, type=int, help='the number of blocks in BrainNetwork')
parser.add_argument('--depth', default=2, type=int, help='the depth in PriorNetwork')
parser.add_argument('--num_epochs', default=150, type=int)
parser.add_argument('--num_sub', default=2,type=int, help='the number of subjects used in the experiments')
parser.add_argument('--batch_size', default=200, type=int, metavar='N', help='batch size')
parser.add_argument('--seed', default=2024, type=int, help='seed for initializing training')
parser.add_argument('--val', default='GA_wo_baseline_corr', type=str)
parser.add_argument('--normalize_features', action='store_true',
                    help='L2 normalize EEG and image features before prior training/inference')

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('param counts:\n{:,} total\n{:,} trainable'.format(total, trainable))
    return trainable


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 26), (1, 1)),
            nn.AvgPool2d((1, 5), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (1024, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # 5 is better than 1
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        x = x.contiguous().view(x.size(0), -1)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))

        
class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1400, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

class channel_attention(nn.Module):
    def __init__(self, sequence_num=200, inter=20):
        super(channel_attention, self).__init__()
        self.sequence_num = sequence_num
        self.inter = inter
        self.extract_sequence = int(self.sequence_num / self.inter)  # You could choose to do that for less computation

        self.query = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),  # also may introduce improvement to a certain extent
            nn.Dropout(0.3)
        )
        self.key = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.Dropout(0.3)
        )

        self.projection = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
        )

        self.drop_out = nn.Dropout(0)
        self.pooling = nn.AvgPool2d(kernel_size=(1, self.inter), stride=(1, self.inter))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        temp = rearrange(x, 'b o c s->b o s c')
        temp_query = rearrange(self.query(temp), 'b o s c -> b o c s')
        temp_key = rearrange(self.key(temp), 'b o s c -> b o c s')

        channel_query = temp_query
        channel_key = temp_key

        scaling = self.extract_sequence ** (1 / 2)

        channel_atten = torch.einsum('b o c s, b o m s -> b o c m', channel_query, channel_key) / scaling

        channel_atten_score = F.softmax(channel_atten, dim=-1)
        channel_atten_score = self.drop_out(channel_atten_score)

        out = torch.einsum('b o c s, b o c m -> b o c s', x, channel_atten_score)
        '''
        projections after or before multiplying with attention score are almost the same.
        '''
        out = rearrange(out, 'b o c s -> b o s c')
        out = self.projection(out)
        out = rearrange(out, 'b o s c -> b o c s')
        return out

from torch_geometric.nn import GATConv
class EEG_GAT(nn.Module):
    def __init__(self, in_channels=200, out_channels=200):
        super(EEG_GAT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = GATConv(in_channels=in_channels, out_channels=out_channels, heads=1)

        self.num_channels = 1024
        # Create a list of tuples representing all possible edges between channels
        self.edge_index_list = torch.Tensor([(i, j) for i in range(self.num_channels) for j in range(self.num_channels) if i != j]).to(device)
        # Convert the list of tuples to a tensor
        self.edge_index = torch.tensor(self.edge_index_list, dtype=torch.long).t().contiguous().to(device)

    def forward(self, x):

        batch_size, _, num_channels, num_features = x.size()
        
        x = x.view(batch_size*num_channels, num_features)
        
        x = self.conv1(x, self.edge_index)
        x = x.view(batch_size, num_channels, -1)
        x = x.unsqueeze(1)
        
        return x


class Enc_eeg(nn.Sequential):
    def __init__(self, input_shape=(1, 1024, 200), emb_size=40, emb_dim=1400, proj_dim=1024):
        super().__init__(
            # Encoder path
            ResidualAdd(
                nn.Sequential(
                    EEG_GAT(),
                    nn.Dropout(0.3),
                )
            ),
            PatchEmbedding(emb_size=emb_size),
            Proj_eeg(embedding_dim=emb_dim, proj_dim=proj_dim)
        )


class EEGToImageModule(nn.Module):
    def __init__(self):
        super(EEGToImageModule, self).__init__()
    def forward(self, x):
        return x

def get_eeg_data(args, nSub):
    train_data = []
    train_label = []
    test_data = []
    test_label = np.arange(100)

    train_data = np.load(args.eeg_data_path + nSub + '/train_data_wo_baseline_corr.npy', allow_pickle=True)
    test_data = np.load(args.eeg_data_path + nSub + '/test_data_wo_basline_corr.npy', allow_pickle=True)
    train_label = np.arange(len(train_data))
    
    return train_data, train_label, test_data, test_label

def get_image_data(args):
    train_img_feature = np.load(os.path.join(args.img_data_path, args.dnn, f'{args.dnn}_feats_train.npy'), allow_pickle=True)
    test_img_feature = np.load(os.path.join(args.img_data_path, args.dnn, f'{args.dnn}_feats_test.npy'), allow_pickle=True)

    return train_img_feature, test_img_feature

def get_eeg_features(train_loader, eeg_model):
    eeg_model.eval()
    eeg_features_list = []
    with torch.no_grad():
        for batch_idx, (eeg_data, _) in enumerate(train_loader):
            eeg_data = eeg_data.to(device).float()
            eeg_features = eeg_model(eeg_data)
            eeg_features_list.append(eeg_features)

    all_eeg_features = torch.cat(eeg_features_list, dim=0).cpu()
    return all_eeg_features

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
    
    for sub in subjects:
        print('The subject is: ', sub)
        eeg_model = Enc_eeg()
        print('The number of parameters of eeg_model:', sum([p.numel() for p in eeg_model.parameters()]))
        model_dir = os.path.join(args.models_root, args.dnn, args.val, sub)
        eeg_model.load_state_dict(torch.load(os.path.join(model_dir, 'Enc_eeg.pth'), weights_only=True), strict=False)
        eeg_model = eeg_model.to(device)
        
        train_data, train_label, test_data, test_label = get_eeg_data(args, sub)
        img_train_features, img_test_features = get_image_data(args)
        print('----------------------------------------------------------------------------------------------------------')
        print('The shape of train data is:', train_data.shape)
        print('The shape of test data is:', test_data.shape)
        test_idx_intrain = np.load('./TVSD/test_idx_intrain.npy')
        train_data = np.delete(train_data, test_idx_intrain, axis=0)
        train_label = np.delete(train_label, test_idx_intrain, axis=0)
        img_train_features = np.delete(img_train_features, test_idx_intrain, axis=0)
        
        train_data = torch.from_numpy(train_data)
        train_label = torch.from_numpy(train_label)
        test_data = torch.from_numpy(test_data.mean(axis=1)).unsqueeze(1)
        print('----------------------------------------------------------------------------------------------------------')
        print('The shape of train data is:', train_data.shape)
        print('The shape of test data is:', test_data.shape)
        print('The shape of img_train_features is:', img_train_features.shape)
        print('The shape of img_test_features is:', img_test_features.shape)
        # get eeg test features
        eeg_model.eval()
        with torch.no_grad():
            test_data = test_data.to(device).float()
            eeg_test_features = eeg_model(test_data)
        eeg_test_features = eeg_test_features.cpu()
        # get eeg train features
        train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
        eeg_train_features = get_eeg_features(train_loader, eeg_model)
        
        print('The shape of eeg_train_features is:', eeg_train_features.shape)
        print('The shape of eeg_test_features is:', eeg_test_features.shape)
        os.makedirs(model_dir, exist_ok=True)
        torch.save({
            'eeg_train_features': eeg_train_features,
        }, os.path.join(model_dir, 'eeg_train_features.pt'))
        torch.save({
            'eeg_test_features': eeg_test_features,
        }, os.path.join(model_dir, 'eeg_test_features.pt'))
        
        eeg_train_features = eeg_train_features.unsqueeze(1)
        img_train_features = torch.from_numpy(img_train_features).float()

        eeg_test_features = eeg_test_features.unsqueeze(1)
        img_test_features = torch.from_numpy(img_test_features).float()

        eeg_train_features = eeg_train_features.float()
        eeg_test_features = eeg_test_features.float()

        
        print('-----------------------------------------------------------------------------------------------------')
        print('The shape of eeg_train_features is:', eeg_train_features.shape)
        print('The shape of img_train_features is:', img_train_features.shape)
        print('The shape of eeg_test_features is:', eeg_test_features.shape)
        print('The shape of img_test_features is:', img_test_features.shape)
        print('-----------------------------------------------------------------------------------------------------')
        print("EEG-to-image embeddings Minimum:", eeg_train_features.min().item())
        print("EEG-to-image embeddings Maximum:", eeg_train_features.max().item())
        print('-----------------------------------------------------------------------------------------------------')
        print("Image embeddings Minimum:", img_train_features.min().item())
        print("Image embeddings Maximum:", img_train_features.max().item())
        print('-----------------------------------------------------------------------------------------------------')
        # shuffle the training data
        train_shuffle = np.random.permutation(len(eeg_train_features))
        eeg_train_features = eeg_train_features[train_shuffle]
        img_train_features = img_train_features[train_shuffle]

        eeg_val_features = eeg_train_features[:748]
        img_val_features = img_train_features[:748]

        eeg_train_features = eeg_train_features[748:]
        img_train_features = img_train_features[748:]

        # Prepare data loaders
        train_dataset = TensorDataset(eeg_train_features, img_train_features)
        val_dataset = TensorDataset(eeg_val_features, img_val_features)
        test_dataset = TensorDataset(eeg_test_features, img_test_features)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        criterion_mse = nn.MSELoss().to(device)
        model = EEGToImageModule()
        # setup diffusion prior network
        timesteps = 100
        prior_network = PriorNetwork(
                dim=args.clip_dim,
                depth=args.depth,
                dim_head=256,
                heads=8,
                causal=False,
                num_tokens=args.num_tokens,
                learned_query_mode="pos_emb"
            ).to(device)

        model.diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=args.clip_dim,
            condition_on_text_encodings=False,
            timesteps=timesteps,
            cond_drop_prob=0.2,
            image_embed_scale=None,
        ).to(device)

        print('The number of params in DiffusionPrior:')
        count_params(model.diffusion_prior)
        print('The number of params in overall model:')
        count_params(model)

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        opt_grouped_parameters = [
            {'params': [p for n, p in model.diffusion_prior.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
            {'params': [p for n, p in model.diffusion_prior.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]

        optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=args.lr)
        total_steps=int(np.floor(args.num_epochs*len(train_loader)))
        print("total_steps", total_steps)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=args.lr,
            total_steps=total_steps,
            final_div_factor=1000,
            last_epoch=-1,
            pct_start=2/args.num_epochs
        )

        # Training loop
        best_val_loss = np.inf
        for epoch in range(args.num_epochs):
            model.train()
            train_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                loss, _ = model.diffusion_prior(text_embed=inputs, image_embed=targets)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                train_loss += loss.item()
            train_loss = train_loss / (batch_idx + 1)
            
            if (epoch + 1) % 1 == 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for idx, (inputs, targets) in enumerate(val_loader):
                        inputs, targets = inputs.to(device), targets.to(device)
                        loss, _ = model.diffusion_prior(text_embed=inputs, image_embed=targets)
                        val_loss += loss.item()
                    val_loss = val_loss / (idx + 1)
                print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(model_dir, 'image_condition.pth'))
                    print("Model saved as image_condition.pth")

        # Testing loop
        model.load_state_dict(torch.load(os.path.join(model_dir, 'image_condition.pth'), weights_only=True), strict=False)
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                prior_out = model.diffusion_prior.p_sample_loop(inputs.shape, text_cond=dict(text_embed=inputs), cond_scale=1.0, timesteps=20)
                loss = criterion_mse(prior_out, targets)
                test_loss += loss.item()
            test_loss = test_loss / (idx + 1)
        print(f"Test Loss: {test_loss}")

        image_model = EEGToImageModule()
        # setup diffusion prior network
        timesteps = 100
        prior_network = PriorNetwork(
                dim=args.clip_dim,
                depth=args.depth,
                dim_head=256,
                heads=8,
                causal=False,
                num_tokens=args.num_tokens,
                learned_query_mode="pos_emb"
            ).to(device)

        image_model.diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=args.clip_dim,
            condition_on_text_encodings=False,
            timesteps=timesteps,
            cond_drop_prob=0.2,
            image_embed_scale=None,
        ).to(device)

        image_model.load_state_dict(torch.load(os.path.join(model_dir, 'image_condition.pth'), weights_only=True), strict=False)
        # Testing loop
        image_model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                prior_out = image_model.diffusion_prior.p_sample_loop(inputs.shape, text_cond=dict(text_embed=inputs), cond_scale=1.0, timesteps=20)
                loss = criterion_mse(prior_out, targets)
                test_loss += loss.item()
            test_loss = test_loss / (idx + 1)
        print(f"After loading, the test Loss: {test_loss}")
        
if __name__ == '__main__':
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))