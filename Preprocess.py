# 把所有的label放到一个文件中
import os
import numpy as np
from PIL import Image
from torchvision import transforms

def load_image(path):
    return Image.open(path).convert('RGB')

DATA_DIR = 'data/garbage_classify/train_data/'
EXPORT_DIR = 'data/'

transforms_dict = {
    True:   transforms.Compose([
        transforms.RandomResizedCrop((200, 200)) ,
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, -.406],[0.229, 0.224, 0.225])]),
    False:  transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, -.406],[0.229, 0.224, 0.225])])
}


data_batch_size = 5
data_aug_size = 5

def get_file_size(aug= False):
    return data_batch_size

def write_to_file(ary, fn):
    ary = np.array(ary)
    np.save(fn, ary)

def main():
    data_num1 = 0
    data_num2 = 0
    total = 0
    data_x_raw = []
    data_y_raw = []
    data_x_aug = []
    data_y_aug = []

    dirs = os.listdir(DATA_DIR)
    for fn in dirs:
        fn_full = os.path.join(DATA_DIR, fn)
        if os.path.isfile(fn_full) and fn.endswith('.txt'):
            with open(fn_full, 'r') as fh:
                ctnts = fh.read()
                ctnts = ctnts.split(',')
                img_full = os.path.join(DATA_DIR, ctnts[0])
                img = load_image(img_full)
                data_x_raw.append(transforms_dict[False](img).numpy().reshape(3, 200, 200))
                data_y_raw.append(int(ctnts[1].strip()))
                for i in range(data_aug_size):
                    data_x_aug.append(transforms_dict[True](img).numpy().reshape(3, 200, 200))
                    data_y_aug.append(int(ctnts[1].strip()))
            total += 1
            if len(data_x_raw) >= data_batch_size:
                write_to_file(data_x_raw, EXPORT_DIR + f'x_{data_num1}.npy')
                data_x_raw = []
                data_num1 += 1
            if len(data_x_aug) >= data_batch_size:
                write_to_file(data_x_aug, EXPORT_DIR + f'xaug_{data_num2}.npy')
                data_x_aug = []
                data_num2 += 1
    np.save(EXPORT_DIR + 'yaug.npy', data_y_aug)
    np.save(EXPORT_DIR + 'y.npy', data_y_raw)

    print(total, total * 5)

if __name__ == '__main__':
    main()