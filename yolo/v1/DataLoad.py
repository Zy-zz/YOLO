import glob
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2

from yolo.v1.util import letterbox

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes


# 数据集
class trainData(Dataset):
    def __init__(self,model_type, data_dir=None):
        """
        data_dir:文件夹位置
        """
        self.data_dir = data_dir
        self.model_type=model_type
        """
        os.path.join:文件路径拼接，多个路径间用‘\’分割
        """
        f = []
        img_list_txt = os.path.join(self.data_dir)  # 储存图片位置的列表
        # label_csv = os.path.join('D:\pycharm\dataset\\trainlabel')  # 储存标签的数组文件
        for p in img_list_txt if isinstance(img_list_txt, list) else [img_list_txt]:
            p = Path(p)  # os-agnostic
            if p.is_dir():  # dir
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)

        # 筛选出图片
        self.im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
        # 筛选出标签
        self.label_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in ['txt'])
        labels_all = []
        for label_i in self.label_files:
            with open(label_i, 'r') as f:
                mid_label = np.loadtxt(f)
                labels_all.append(mid_label.tolist())

        self.label = labels_all

        self.num_all_data = len(self.im_files)

    def __len__(self):
        """获取数据集数量"""
        return self.num_all_data

    def __getitem__(self, idx):
        # trans = transforms.Compose([
        #     # transforms.Resize((112,112)),
        #     transforms.ToTensor(requires_grad=True),
        # ])
        img_path = self.im_files[idx]
        img = cv2.imread(img_path)
        if self.model_type=="v1":
            newsize=448
            outputsize=7
        elif self.model_type=="v2":
            newsize = 416
            outputsize=13
        img = letterbox(img, new_shape=(newsize, newsize), auto=False, scaleFill=True)
        label = torch.tensor(self.label[idx])
        # 制作训练标签
        label = convert_bbox2labels(label,outputsize)
        return torch.tensor(img[0].astype(float)).permute(2, 0, 1).float(), label


# 加载测试集数据集
def testData(data_dir):
    f = []
    img_list_txt = os.path.join(data_dir)  # 储存图片位置的列表
    # label_csv = os.path.join('D:\pycharm\dataset\\trainlabel')  # 储存标签的数组文件
    for p in img_list_txt if isinstance(img_list_txt, list) else [img_list_txt]:
        p = Path(p)  # os-agnostic
        if p.is_dir():  # dir
            f += glob.glob(str(p / '**' / '*.*'), recursive=True)

    # 筛选出图片
    im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
    # 筛选出标签
    label_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in ['txt'])
    labels_all = []
    for label_i in label_files:
        with open(label_i, 'r') as f:
            mid_label = np.loadtxt(f)
            labels_all.append(mid_label.tolist())

    return im_files, labels_all


def generate_dxdywh(gt_label, w, h, s):
    xmin, ymin, xmax, ymax = gt_label[:-1]
    # 计算边界框的中心点
    c_x = (xmax + xmin) / 2 * w
    c_y = (ymax + ymin) / 2 * h
    box_w = (xmax - xmin) * w
    box_h = (ymax - ymin) * h

    if box_w < 1e-4 or box_h < 1e-4:
        # print('Not a valid data !!!')
        return False

    # 计算中心点所在的网格坐标
    c_x_s = c_x / s
    c_y_s = c_y / s
    grid_x = int(c_x_s)
    grid_y = int(c_y_s)
    # 计算中心点偏移量和宽高的标签
    tx = c_x_s - grid_x
    ty = c_y_s - grid_y
    tw = np.log(box_w)
    th = np.log(box_h)
    # 计算边界框位置参数的损失权重
    weight = 2.0 - (box_w / w) * (box_h / h)

    return grid_x, grid_y, tx, ty, tw, th, weight


def convert_bbox2labels(bbox,output_w):
    """将bbox的(cls,x,y,w,h)数据转换为训练时方便计算Loss的数据形式(7,7,5*B+cls_num)
        注意，输入的bbox的信息是(xc,yc,w,h)格式的，转换为labels后，bbox的信息转换为了(px,py,w,h)格式"""
    gridsize = 1.0 / output_w
    labels = torch.zeros([output_w, output_w, 1 + 1 + 4])  # 注意，此处需要根据不同数据集的类别个数进行修改
    # 这里无视batch，默认1

    if len(bbox.shape) > 1:
        for k in range(bbox.shape[0]):  # 要判断传进来的label里有多少个框
            for i in range(len(bbox[k]) // 5):
                gridx = int(bbox[k][i * 5 + 1] / gridsize)  # 当前bbox中心落在第gridx个网格,列
                gridy = int(bbox[k][i * 5 + 2] / gridsize)  # 当前bbox中心落在第gridy个网格,行
                # 计算中心点偏移量和宽高的标签
                gridpx = bbox[k][i * 5 + 1] - gridx * gridsize
                gridpy = bbox[k][i * 5 + 2] - gridy * gridsize

                # 计算边界框位置参数的损失权重
                # weight = 2.0 - bbox[k][3]* bbox[k][4]
                # 将第gridy行，gridx列的网格设置为负责当前ground truth的预测，置信度和对应类别概率均置为1
                labels[gridy, gridx, 0:6] = torch.tensor([1, bbox[k][0], gridpx, gridpy, bbox[k][3], bbox[k][4]])
    else:
        for i in range(len(bbox) // 5):
            gridx = int(bbox[i * 5 + 1] / gridsize)  # 当前bbox中心落在第gridx个网格,列
            gridy = int(bbox[i * 5 + 2] / gridsize)  # 当前bbox中心落在第gridy个网格,行
            # (bbox中心坐标 - 网格左上角点的坐标)/网格大小  ==> bbox中心点的相对位置
            gridpx = bbox[i * 5 + 1] - gridx * gridsize
            gridpy = bbox[i * 5 + 2] - gridy * gridsize

            # 计算边界框位置参数的损失权重
            # weight = 2.0 - bbox[3] * bbox[4]
            # 将第gridy行，gridx列的网格设置为负责当前ground truth的预测，置信度和对应类别概率均置为1
            labels[gridy, gridx, 0:6] = torch.tensor([1, bbox[0], gridpx, gridpy, bbox[3], bbox[4]])

    labels = labels.reshape(-1, 1 + 1 + 4)
    return labels


if __name__ == '__main__':
    dataset_dir = r"D:\pycharm\dataset"
    traindata = trainData(dataset_dir)
    dataloader = DataLoader(traindata, 1)
    for i, (imgs, labels) in enumerate(dataloader):
        labels = convert_bbox2labels(labels,7)
        labels = torch.tensor(labels)
        labels = labels.view(1, 7, 7, -1)
        labels = labels.permute(0, 3, 1, 2)
