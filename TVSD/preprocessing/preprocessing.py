"""
preprocessing the THINGS Ventral Stream Dataset (TVSD)
"""
import argparse
import os
import numpy as np
import h5py

sub = ['monkeyF', 'monkeyN']
parser = argparse.ArgumentParser(description='Preprocess TVSD neural recordings')
parser.add_argument('--data_root', default='./data/TVSD/', type=str)
parser.add_argument('--save_path', default='./data/TVSD/Preprocessed_data/', type=str)
args = parser.parse_args()

file_path = args.data_root
save_path = args.save_path
# load the data

for s in sub:
    train_data = np.zeros((22248, 1, 1024, 300))
    test_data = np.zeros((100, 30, 1024, 300))
    data_path = os.path.join(file_path, s, 'THINGS_MUA_trials.mat')
    with h5py.File(data_path, 'r') as f:
        data_sub = f['ALLMUA']
        data_sub = np.array(data_sub)
        data_sub = data_sub.transpose(1, 2, 0) # (trials, channels, time)
        info_sub = f['ALLMAT']    
        train_idx_sub = info_sub[1]
        test_idx_sub = info_sub[2]  
        for i in range(len(train_data)):
            train_data[i] = data_sub[np.where(train_idx_sub == i+1)[0]]
        for i in range(len(test_data)):
            test_data[i] = data_sub[np.where(test_idx_sub == i+1)[0]]
    # preprocess the data
    for i in range(len(train_data)):
        for j in range(len(train_data[i, 0])):
            train_data[i, 0, j] = train_data[i, 0, j] - np.mean(train_data[i, 0, j, :100], axis=-1)
    for i in range(len(test_data)):
        for j in range(len(test_data[i])):
            for k in range(len(test_data[i, j])):
                test_data[i, j, k] = test_data[i, j, k] - np.mean(test_data[i, j, k, :100], axis=-1)

    train_data = train_data[:, :, :, 100:]
    test_data = test_data[:, :, :, 100:]

    # z-score
    for i in range(len(train_data)):
        train_data[i] = (train_data[i] - np.mean(train_data[i])) / np.std(train_data[i])
    for i in range(len(test_data)):
        for j in range(len(test_data[i])):
            test_data[i][j] = (test_data[i][j] - np.mean(test_data[i][j])) / np.std(test_data[i][j])
    # save the data 
    save_path_sub = os.path.join(save_path, s)
    if not os.path.exists(save_path_sub):
        os.makedirs(save_path_sub)
    train_data_path = os.path.join(save_path_sub, 'train_data_w_baseline_corr.npy')
    test_data_path = os.path.join(save_path_sub, 'test_data_w_basline_corr.npy')
    np.save(train_data_path, train_data)
    np.save(test_data_path, test_data)
            
print('111')
            


        
