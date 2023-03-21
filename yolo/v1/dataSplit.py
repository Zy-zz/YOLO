import glob
import os
import shutil
from pathlib import Path

import numpy as np

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes

img_list_txt = os.path.join(r"D:\pycharm\datasets\voc2007\trainimg")  # 储存图片位置的列表
label_list_txt = os.path.join(r"D:\pycharm\datasets\voc2007\trainlabel")  # 储存图片位置的列表
f = []
for p in img_list_txt if isinstance(img_list_txt, list) else [img_list_txt]:
    p = Path(p)  # os-agnostic
    if p.is_dir():  # dir
        f += glob.glob(str(p / '**' / '*.*'), recursive=True)

for p in label_list_txt if isinstance(label_list_txt, list) else [label_list_txt]:
    p = Path(p)  # os-agnostic
    if p.is_dir():  # dir
        f += glob.glob(str(p / '**' / '*.*'), recursive=True)

# 筛选出图片
im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
# 筛选出标签
label_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in ['txt'])

for i in range(1000):
    targetName = im_files[i].split("\\")[-1][:-4]
    whole_label_name = r"D:/pycharm/datasets/voc2007/trainlabel/" + targetName + '.txt'
    whole_img_name = r"D:/pycharm/datasets/voc2007/trainimg/" + targetName + '.jpg'
    if os.path.exists(whole_label_name):
        shutil.copy(whole_img_name, r'C:\Users\ZY\Desktop\新建文件夹\trainimg')
        shutil.copy(whole_label_name, r'C:\Users\ZY\Desktop\新建文件夹\trainlabel')
