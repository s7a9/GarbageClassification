import torch
from torch import nn, optim
from torchvision import transforms
import os
import logging
import argparse
from gcDataLoader import get_loaders
from ParseConstants import *
# 设置log
logging.basicConfig(level=logging.INFO, 
    format='%(asctime)s %(filename)s:%(lineno)d [%(levelname)s] %(message)s')
# 设置参数解析
parser = argparse.ArgumentParser(description= 'Train a garbage classification model.')
parser.add_argument('--model', type= str, default= 'resnet18', 
    help= 'Choose model (from resnet18(default) resnet50 resnet101)')
parser.add_argument('--bs', type= int, default= 16,
    help= 'Batch size (default 16)')
parser.add_argument('-aug', action= 'store_const', 
    const= True, default= False,
    help= 'Use image augmentation')
parser.add_argument('--lr', type= float, default= 1e-4,
    help= 'Learning rate (default 1e-4)')
parser.add_argument('--optim', type= str, default= 'SGD',
    help= 'Choose optimizer (SGD(default) Adam Adadelta)')
parser.add_argument('--scheduler', type= str, default= 'None',
    help= 'Chose scheduler (StepLR ExpLr CosLR Plateau)')
parser.add_argument('--epoch', type= int, default= 20,
    help= 'Number of Epoches (defualt 20)')

args = parser.parse_args()

BATCH_SIZE = args.bs

logging.info('begin to load data...')
train_loader, test_loader = get_loaders(0.8, BATCH_SIZE, args.aug)
train_len = len(train_loader.dataset)
test_len = len(test_loader.dataset)
logging.info(f'size of training set: {train_len}')
logging.info(f'size of testing set: {test_len}')

device = torch.device('cuda')
# model = ResNet18().to(device)
model = model_dict[args.model]
num_in = model.fc.in_features
model.fc = nn.Linear(num_in, 40)
model = model.to(device)
print(model, flush= True)

use_scheduler = args.scheduler != 'None'
if use_scheduler:
    optimizer, scheduler = get_optimizer(model, args.lr, args.scheduler, args.optim)
else:
    optimizer = get_optimizer(model, args.lr, args.scheduler, args.optim)
# optimizer = optim.SGD(model.parameters(), lr=1e-4)  #TODO
criteon = nn.CrossEntropyLoss()

logging.info('begin to train...')
for epoch in range(args.epoch):
    train_loss = 0
    test_loss = 0

    model.train()
    for x, label in train_loader:
        x, label = x.to(device), label.to(device)
        logits = model(x)
        loss = criteon(logits, label)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if use_scheduler:
            scheduler.step()

    model.eval()
    with torch.no_grad():
        total_corret = 0
        total_num = 0
        for x, label in test_loader:
            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = criteon(logits, label)
            test_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct = torch.eq(pred, label).float().sum().item()
            total_corret += correct
            total_num += x.size(0)
        acc = total_corret / total_num

    logging.info(f'[{epoch}] train_loss: {train_loss / train_len}, test_loss: {test_loss / test_len}, test_accuracy: {acc}')
