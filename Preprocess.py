# 把所有的label放到一个文件中
import os

DATA_DIR = 'data/garbage_classify/train_data/'
EXPORT_PATH = 'data/garbage_classify/labels.txt'

fh_out = open(EXPORT_PATH, 'w+')

dirs = os.listdir(DATA_DIR)

for fn in dirs:
    fn_full = os.path.join(DATA_DIR, fn)
    if os.path.isfile(fn_full) and fn.endswith('.txt'):
        with open(fn_full, 'r') as fh:
            ctnts = fh.read()
            ctnts = ctnts.split(',')
            fh_out.write(ctnts[0] + ctnts[1] + '\n')

fh_out.close()