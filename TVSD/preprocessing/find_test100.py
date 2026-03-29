"""
discard the test 100 conditions from the training set (get the position from 22248)
"""
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Find test-condition indices within train CSV')
parser.add_argument('--train_csv', default='./TVSD/things_imgs_train.csv', type=str)
parser.add_argument('--test_csv', default='./TVSD/things_imgs_test.csv', type=str)
parser.add_argument('--save_path', default='./TVSD/test_idx_intrain.npy', type=str)
args = parser.parse_args()

condi_path = pd.read_csv(args.train_csv)
condi_path = condi_path.iloc[:, 0]
# get the condition before '\'
condi_path_con = condi_path.str.split('\\').str[0]

# save into list
condi_path_test = pd.read_csv(args.test_csv)
condi_path_test = condi_path_test.iloc[:, 0]
# choose the test condition before '\'
condi_path_test_con = condi_path_test.str.split('\\').str[0]

# get the index of the test conditions in the training set
idx = []
for i in range(len(condi_path_con)):
    if condi_path_con[i] in condi_path_test_con.values:
        idx.append(i)

idx = np.array(idx, dtype=np.int64)
np.save(args.save_path, idx)
print(f'Saved {len(idx)} indices to {args.save_path}')