from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
import numpy as np

import os
import logging
import math

LABEL_PATH = 'data/garbage_classify/labels.txt'
DATA_DIR = 'data/garbage_classify/train_data/'

def load_image(path):
    return Image.open(path).convert('RGB')
    
class MyDataset(Dataset):
    def __init__(self, transform):
        self.transform = transform
        self.data_info = []
        with open(LABEL_PATH, 'r') as fh:
            for line in fh.readlines():
                line = line.split(' ')
                self.data_info.append((line[0], int(line[1])))
    
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, index):
        fn, lbl = self.data_info[index]
        data = self.transform(load_image(os.path.join(DATA_DIR, fn)))
        return data, lbl


def get_loaders(train_test_ratio, batch_size, transform):
    full_set = MyDataset(transform)
    trainning_size = math.ceil(len(full_set) * train_test_ratio)
    train_set, test_set = random_split(dataset= full_set, 
        lengths= [trainning_size, len(full_set) - trainning_size])
    logging.info(f'size of full_data: {len(full_set)}, train_data: {trainning_size}')
    train_loader = DataLoader(train_set, batch_size= batch_size, shuffle= True)
    test_loader  = DataLoader(test_set,  batch_size= batch_size, shuffle= True)
    return train_loader, test_loader

# from torchvision import transforms

# if __name__ == '__main__':
#     xxx, yyy= get_loaders(0, 16,
#         transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomVerticalFlip(),
#             transforms.RandomRotation(90),
#             transforms.RandomRotation(90),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, -.406],[0.229, 0.224, 0.225])]))
#     d, l = xxx[0]
#     print(d)
