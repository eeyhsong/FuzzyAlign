"""
Fuzzy Alignment
"""

import os

gpus = [2]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

import argparse
import math
import random
import itertools
import datetime
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor

from einops import rearrange
from einops.layers.torch import Rearrange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Experiment Stimuli Recognition test with CLIP encoder')
parser.add_argument('--result_path', default='./results/', type=str)
parser.add_argument('--eeg_data_path', default='./data/TVSD/Preprocessed_data/', type=str)
parser.add_argument('--img_data_root', default='./data/TVSD/DNN_feature_maps/', type=str)
parser.add_argument('--dnn', default='clip_h14', type=str, help="eva or clip_h14")
parser.add_argument('--epoch', default='50', type=int)
parser.add_argument('--num_sub', default=2, type=int, help='number of subjects used in the experiments. ')
parser.add_argument('-batch_size', '--batch-size', default=150, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=2026, type=int, help='seed for initializing training. ')
parser.add_argument('--val', default='GA_wo_baseline_corr', type=str)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


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
        self.edge_index_list = torch.Tensor([(i, j) for i in range(self.num_channels) for j in range(self.num_channels) if i != j]).to(device)
        self.edge_index = torch.tensor(self.edge_index_list, dtype=torch.long).t().contiguous().to(device)

    def forward(self, x):
        batch_size, _, num_channels, num_features = x.size()

        x = x.view(batch_size*num_channels, num_features)

        x = self.conv1(x, self.edge_index)
        x = x.view(batch_size, num_channels, -1)
        x = x.unsqueeze(1)

        return x

class Enc_eeg(nn.Module):
    def __init__(self, input_shape=(1, 1024, 200), emb_size=40, emb_dim=1400, proj_dim=1024):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(emb_size=emb_size)
        self.proj_eeg = Proj_eeg(embedding_dim=emb_dim, proj_dim=proj_dim)
        self.res_gat = ResidualAdd(
            nn.Sequential(
                EEG_GAT(),
                nn.Dropout(0.3),
            )
        )
    
    def forward(self, x):
        x = self.res_gat(x)
        x_embedded = self.patch_embed(x)
        x_projected = self.proj_eeg(x_embedded)
        return x_projected
    
class FuzzyAlignmentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def fuzzy_similarity(self, x, y):
        """Compute fuzzy similarity using triangular norms"""
        x_norm = F.normalize(x, dim=1)
        y_norm = F.normalize(y, dim=1)

        similarity = torch.mm(x_norm, y_norm.t())

        fuzzy_sim = torch.sigmoid(similarity / self.temperature)

        return fuzzy_sim
    
    def forward(self, eeg_features, img_features):
        fuzzy_sim = self.fuzzy_similarity(eeg_features, img_features)

        batch_size = eeg_features.shape[0]
        labels = torch.arange(batch_size).cuda()

        loss = F.cross_entropy(fuzzy_sim / self.temperature, labels)
        return loss

class IE():
    def __init__(self, args, nsub):
        super(IE, self).__init__()
        self.args = args
        self.num_class = 100
        self.batch_size = args.batch_size
        self.result_path = args.result_path
        self.batch_size_test = 100
        self.n_epochs = args.epoch
        self.val = args.val
        self.dnn = args.dnn

        self.lambda_cen = 0.003
        self.alpha = 0.5

        self.proj_dim = 256

        self.lr = 0.001
        self.b1 = 0.5
        self.b2 = 0.999
        self.nSub = nsub

        self.start_epoch = 0
        self.eeg_data_path = args.eeg_data_path
        self.img_data_path = os.path.join(args.img_data_root, self.dnn) + '/'
        self.test_center_path = os.path.join(args.img_data_root, self.dnn) + '/'
        self.pretrain = False
        
        os.makedirs(self.result_path + f"{self.dnn}/{self.val}/", exist_ok=True) 
        self.log_write = open(self.result_path + f"{self.dnn}/{self.val}/" + "log_subject%d.txt" % self.nSub, "w")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().to(device)
        self.criterion_l2 = torch.nn.MSELoss().to(device)
        self.criterion_cls = torch.nn.CrossEntropyLoss().to(device)
        self.criterion_fuzzy = FuzzyAlignmentLoss().to(device)

        self.Enc_eeg = Enc_eeg().to(device)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.centers = {}
        print('initial define done.')

    def get_eeg_data(self):
        train_data = []
        train_label = []
        test_data = []
        test_label = np.arange(100)

        train_data = np.load(self.eeg_data_path + '/sub-' + format(self.nSub, '02') + '/train_data_wo_baseline_corr.npy', allow_pickle=True)
        test_data = np.load(self.eeg_data_path + '/sub-' + format(self.nSub, '02') + '/test_data_wo_basline_corr.npy', allow_pickle=True)
        
        return train_data, train_label, test_data, test_label

    def get_image_data(self):
        train_img_feature = np.load(self.img_data_path + self.args.dnn + '_feats_train.npy', allow_pickle=True)
        test_img_feature = np.load(self.img_data_path + self.args.dnn + '_feats_test.npy', allow_pickle=True)

        train_img_feature = np.squeeze(train_img_feature)
        test_img_feature = np.squeeze(test_img_feature)
        
        return train_img_feature, test_img_feature
    
    def select_image(self, image, label):
        select_image = []
        for i in range(self.num_class):
            catgory_idx = np.where(label == i)
            select_idx = np.random.choice(catgory_idx[0], 180, replace=False)
            select_image.append(image[select_idx])
        select_image = torch.cat(select_image)
        return select_image

    def train(self):
        train_eeg, _, test_eeg_all, test_label = self.get_eeg_data()
        train_img_feature, test_img_feature = self.get_image_data()
        test_idx_intrain = np.load('./TVSD/test_idx_intrain.npy')
        train_eeg = np.delete(train_eeg, test_idx_intrain, axis=0)
        train_img_feature = np.delete(train_img_feature, test_idx_intrain, axis=0)

        test_center = np.load(self.test_center_path + 'center_' + self.args.dnn + '.npy', allow_pickle=True)
        print('The shape of test_center is: ', test_center.shape)
        # shuffle the training data
        train_shuffle = np.random.permutation(len(train_eeg))
        train_eeg = train_eeg[train_shuffle]
        train_img_feature = train_img_feature[train_shuffle]

        val_eeg = torch.from_numpy(train_eeg[:748])
        val_image = torch.from_numpy(train_img_feature[:748])

        train_eeg = torch.from_numpy(train_eeg[748:])
        train_image = torch.from_numpy(train_img_feature[748:])
        
        print('The shape of train_eeg is: ', train_eeg.shape)
        print('The shape of train_image is: ', train_image.shape)
        print('The shape of val_eeg is: ', val_eeg.shape)
        print('The shape of val_image is: ', val_image.shape)

        dataset = torch.utils.data.TensorDataset(train_eeg, train_image)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = torch.utils.data.TensorDataset(val_eeg, val_image)
        self.val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)

        test_eeg = torch.from_numpy(test_eeg_all.mean(axis=1).reshape(-1, 1, 1024, 200))
        
        print('The shape of test_eeg is: ', test_eeg.shape)
        print('The shape of test_img_feature is: ', test_img_feature.shape)
        test_center = torch.from_numpy(test_center)
        test_label = torch.from_numpy(test_label)
        test_dataset = torch.utils.data.TensorDataset(test_eeg, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test, shuffle=False)
        self.optimizer = torch.optim.Adam(itertools.chain(self.Enc_eeg.parameters()), lr=self.lr, betas=(self.b1, self.b2))
        del train_eeg, train_img_feature, val_eeg, val_image

        best_loss_val = np.inf

        for e in range(self.n_epochs):
            self.Enc_eeg.train()

            for i, (eeg, img) in enumerate(self.dataloader):

                eeg = eeg.to(device).type(self.Tensor)
                img_features = img.to(device).type(self.Tensor)
                labels = torch.arange(eeg.shape[0])
                labels = labels.to(device).type(self.LongTensor)

                eeg_features = self.Enc_eeg(eeg)
                loss_dis = self.criterion_l2(eeg_features, img_features)

                eeg_features = eeg_features / eeg_features.norm(dim=1, keepdim=True)
                img_features = img_features / img_features.norm(dim=1, keepdim=True)

                logit_scale = self.logit_scale.exp()
                logits_per_eeg = logit_scale * eeg_features @ img_features.t()
                logits_per_img = logits_per_eeg.t()

                loss_eeg = self.criterion_cls(logits_per_eeg, labels)
                loss_img = self.criterion_cls(logits_per_img, labels)

                loss_con = (loss_eeg + loss_img) / 2
                loss = loss_con + loss_dis

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


            if (e + 1) % 1 == 0:
                self.Enc_eeg.eval()
                vlosses = []
                with torch.no_grad():
                    for i, (veeg, vimg) in enumerate(self.val_dataloader):

                        veeg = veeg.to(device).type(self.Tensor)
                        vimg_features = vimg.to(device).type(self.Tensor)
                        vlabels = torch.arange(veeg.shape[0])
                        vlabels = vlabels.to(device).type(self.LongTensor)

                        veeg_features = self.Enc_eeg(veeg)

                        vloss_dis = self.criterion_l2(veeg_features, vimg_features)

                        veeg_features = veeg_features / veeg_features.norm(dim=1, keepdim=True)
                        vimg_features = vimg_features / vimg_features.norm(dim=1, keepdim=True)

                        logit_scale = self.logit_scale.exp()
                        vlogits_per_eeg = logit_scale * veeg_features @ vimg_features.t()
                        vlogits_per_img = vlogits_per_eeg.t()

                        vloss_eeg = self.criterion_cls(vlogits_per_eeg, vlabels)
                        vloss_img = self.criterion_cls(vlogits_per_img, vlabels)

                        vloss_con = (vloss_eeg + vloss_img) / 2

                        vloss = vloss_con + vloss_dis
                        vlosses.append(vloss)
                    vlosses = torch.stack(vlosses).mean()

                    if vlosses <= best_loss_val:
                        best_loss_val = vlosses
                        best_epoch = e + 1
                        os.makedirs(f"./models/{self.dnn}/{self.val}/sub-0{self.nSub}/", exist_ok=True)
                        torch.save(self.Enc_eeg.state_dict(), f'./models/{self.dnn}/{self.val}/sub-0{self.nSub}/Enc_eeg.pth')

                print('Epoch:', e,
                      '  tcon: %.4f' % loss_eeg.detach().cpu().numpy(),
                      '  vcon: %.4f' % vloss_eeg.detach().cpu().numpy(),
                      '  tdis: %.4f' % loss_dis.detach().cpu().numpy(),
                      '  vdis: %.4f' % vloss_dis.detach().cpu().numpy(),
                      )
                self.log_write.write('Epoch %d: tcon: %.4f, vcon: %.4f, tdis: %.4f, vdis: %.4f\n' % (e, loss_eeg.detach().cpu().numpy(), vloss_eeg.detach().cpu().numpy(), loss_dis.detach().cpu().numpy(), vloss_dis.detach().cpu().numpy()))

            torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        all_center = test_center
        total = 0
        top1 = 0
        top3 = 0
        top5 = 0
        top1_retrieval = 0
        top3_retrieval = 0
        top5_retrieval = 0
        total_mse = []
        total_mse_opt = []
        total_mae = []
        total_mae_opt = []
        total_rho = []
        total_rho_opt = []
        total_mse_all_trial = []
        total_mae_all_trial = []
        total_rho_all_trial = []
        total_rho_all_trial_opt = []

        self.Enc_eeg.load_state_dict(torch.load(f'./models/{self.dnn}/{self.val}/sub-0{self.nSub}/Enc_eeg.pth'), strict=False)
        self.Enc_eeg.eval()

        test_img_feature = torch.from_numpy(test_img_feature).to(device)
        test_img_feature = test_img_feature.type(self.Tensor)

        with torch.no_grad():
            for i, (teeg, tlabel) in enumerate(self.test_dataloader):
                teeg = teeg.to(device).type(self.Tensor)
                tlabel = tlabel.to(device).type(self.LongTensor)
                all_center = all_center.to(device).type(self.Tensor)

                tfea = self.Enc_eeg(teeg)

                tfea = tfea / tfea.norm(dim=1, keepdim=True)
                similarity = (100.0 * tfea @ all_center.t()).softmax(dim=-1)
                _, indices = similarity.topk(5)

                tt_label = tlabel.view(-1, 1)
                total += tlabel.size(0)
                top1 += (tt_label == indices[:, :1]).sum().item()
                top3 += (tt_label == indices[:, :3]).sum().item()
                top5 += (tt_label == indices).sum().item()

                test_img_feature_norm = test_img_feature / test_img_feature.norm(dim=1, keepdim=True)
                similarity_retrieval = (100.0 * tfea @ test_img_feature_norm.t()).softmax(dim=-1)
                _, indices_retrieval = similarity_retrieval.topk(5)
                top1_retrieval += (tt_label == indices_retrieval[:, :1]).sum().item()
                top3_retrieval += (tt_label == indices_retrieval[:, :3]).sum().item()
                top5_retrieval += (tt_label == indices_retrieval).sum().item()


            top1_acc = float(top1) / float(total)
            top3_acc = float(top3) / float(total)
            top5_acc = float(top5) / float(total)

            top1_retrieval_acc = float(top1_retrieval) / float(total)
            top3_retrieval_acc = float(top3_retrieval) / float(total)
            top5_retrieval_acc = float(top5_retrieval) / float(total)

            total_mse = np.mean(total_mse)
            total_mae = np.mean(total_mae)
            total_mse_opt = np.mean(total_mse_opt)
            total_mae_opt = np.mean(total_mae_opt)
            total_rho = np.mean(total_rho)
            total_rho_opt = np.mean(total_rho_opt)
            total_mse_all_trial = np.mean(total_mse_all_trial)
            total_mae_all_trial = np.mean(total_mae_all_trial)
            total_rho_all_trial = np.mean(total_rho_all_trial)
            total_rho_all_trial_opt = np.mean(total_rho_all_trial_opt)


        print('Best epoch: %d' % best_epoch)
        print('The test Top1-%.6f, Top3-%.6f, Top5-%.6f' % (top1_acc, top3_acc, top5_acc))
        print('The retrieval test Top1-%.6f, Top3-%.6f, Top5-%.6f' % (top1_retrieval_acc, top3_retrieval_acc, top5_retrieval_acc))
        self.log_write.write('The best epoch is: %d\n' % best_epoch)
        self.log_write.write('The test Top1-%.6f, Top3-%.6f, Top5-%.6f\n' % (top1_acc, top3_acc, top5_acc))
        self.log_write.write('The retrieval test Top1-%.6f, Top3-%.6f, Top5-%.6f\n' % (top1_retrieval_acc, top3_retrieval_acc, top5_retrieval_acc))

        torch.cuda.empty_cache()

        return top1_acc, top3_acc, top5_acc, top1_retrieval_acc, top3_retrieval_acc, top5_retrieval_acc, total_mse, total_mse_opt, total_mae, total_mae_opt, total_rho, total_rho_opt, total_mse_all_trial, total_mae_all_trial, total_rho_all_trial, total_rho_all_trial_opt


def main():
    args = parser.parse_args()

    num_sub = args.num_sub
    aver = []
    aver3 = []
    aver5 = []
    aver_retri = []
    aver3_retri = []
    aver5_retri = []

    for i in range(num_sub):
        starttime = datetime.datetime.now()
        seed_n = np.random.randint(args.seed)
        seed_n = 42

        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)
        print('Subject %d' % (i+1))
        ie = IE(args, i + 1)

        Acc, Acc3, Acc5, Acc_retri, Acc3_retri, Acc5_retri, mse, mse_opt, mae, mae_opt, rho, rho_opt, mse_all, mae_all, rho_all, rho_all_opt = ie.train()
        print('THE BEST ACCURACY IS ' + str(Acc))
        endtime = datetime.datetime.now()
        print('subject %d duration: '%(i+1) + str(endtime - starttime))

        aver.append(Acc)
        aver3.append(Acc3)
        aver5.append(Acc5)
        aver_retri.append(Acc_retri)
        aver3_retri.append(Acc3_retri)
        aver5_retri.append(Acc5_retri)

    aver.append(np.mean(aver))
    aver3.append(np.mean(aver3))
    aver5.append(np.mean(aver5))
    aver_retri.append(np.mean(aver_retri))
    aver3_retri.append(np.mean(aver3_retri))
    aver5_retri.append(np.mean(aver5_retri))

    aver.append(np.std(aver))
    aver3.append(np.std(aver3))
    aver5.append(np.std(aver5))
    aver_retri.append(np.std(aver_retri))
    aver3_retri.append(np.std(aver3_retri))
    aver5_retri.append(np.std(aver5_retri))

    aver = [round(i, 4) for i in aver]
    aver3 = [round(i, 4) for i in aver3]
    aver5 = [round(i, 4) for i in aver5]
    aver5_retri = [round(i, 4) for i in aver5_retri]
    aver3_retri = [round(i, 4) for i in aver3_retri]
    aver_retri = [round(i, 4) for i in aver_retri]

    column = np.arange(1, num_sub+1)
    column = np.append(column, ['mean', 'std'])
    pd_all = pd.DataFrame([column, aver, aver3, aver5, aver_retri, aver3_retri, aver5_retri], 
                          index=['sub', 'top1', 'top3', 'top5', 'top1-retri', 'top3-retri', 'top5-retri'])
    
    pd_all.to_csv(args.result_path + f'{args.dnn}/{args.val}/result.csv')


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))
