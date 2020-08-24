from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
import torch

import os
import logging
import math
import numpy as np
from Preprocess import get_file_size, EXPORT_DIR, data_aug_size

LABEL_PATH = 'data/'

DATA_SIZE = 14800

class MyDataset(Dataset):
    def __init__(self, aug= False):
        self.is_augmented = aug
        self.file_batch_size = get_file_size(aug)
        self.data_x_path = os.path.join(EXPORT_DIR, 'xaug_' if aug else 'x_')
        self.data_y = np.load(os.path.join(EXPORT_DIR, 'yaug.npy' if aug else 'y.npy'))
        self.data_size = DATA_SIZE * data_aug_size if aug else DATA_SIZE
    
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, index):
        file_index = index // self.file_batch_size
        batch_index = index % self.file_batch_size
        X = torch.from_numpy(np.load(self.data_x_path + str(file_index) + '.npy')[batch_index])
        y = self.data_y[index]
        y = torch.tensor(y, dtype= torch.long)
        return X, y


def get_loaders(train_test_ratio, batch_size, is_augmented):
    full_set = MyDataset(is_augmented)
    trainning_size = math.ceil(len(full_set) * train_test_ratio)
    train_set, test_set = random_split(dataset= full_set, 
        lengths= [trainning_size, len(full_set) - trainning_size])
    logging.info(f'size of full_data: {len(full_set)}, train_data: {trainning_size}')
    train_loader = DataLoader(train_set, batch_size= batch_size, shuffle= True)
    test_loader  = DataLoader(test_set,  batch_size= batch_size, shuffle= True)
    return train_loader, test_loader

if __name__ == '__main__':
    full_set = MyDataset(True)
    X, y = full_set[15]
    print(X.shape, y.shape)
    # print(X, y)
