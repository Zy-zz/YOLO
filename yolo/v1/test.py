###############
import torch

from yolo.v1.DataLoad import convert_bbox2labels


def decode_boxes(pred):
    """
        将txtytwth转换为常用的x1y1x2y2形式。
    """
    output = torch.zeros_like(pred)
    # # 得到所有bbox 的中心点坐标和宽高
    # pred[..., :2] = torch.sigmoid(pred[..., :2]) + self.create_grid(7) # 加上坐标
    # pred[..., 2:] = torch.exp(pred[..., 2:])
    pred[..., :2] = (pred[..., :2] * 13 + create_grid(13)) / 13  # 加上坐标
    # 将所有bbox的中心带入坐标和宽高换算成x1y1x2y2形式
    output[..., :2] = pred[..., :2] - pred[..., 2:] * 0.5
    output[..., 2:] = pred[..., :2] + pred[..., 2:] * 0.5

    return output

def create_grid(input_size):
    """
        用于生成G矩阵，其中每个元素都是特征图上的像素坐标。
    """
    # 输入图像的宽和高
    w, h = input_size, input_size
    # 特征图的宽和高
    ws, hs = w, h
    # 生成网格的x坐标和y坐标
    grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])

    # 将xy两部分的坐标拼起来：[H, W, 2]
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
    # [H, W, 2] -> [HW, 2] -> [HW, 2]
    grid_xy = grid_xy.view(-1, 2)

    return grid_xy


import numpy as np
import matplotlib.pyplot as plt
import cv2
image = cv2.imread(r"D:\pycharm\datasets\voc2007\trainimg\000005.jpg") # 打开一张图片
label=np.loadtxt(r"D:\pycharm\datasets\voc2007\trainlabel\000005.txt")
label_after=convert_bbox2labels(label,13)
label_after_txtypwph=label_after[...,2:]
label_after[...,2:]=decode_boxes(label_after_txtypwph)
label_after_after=np.array(label_after)

py,px,_=image.shape
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
for nowlabel in label_after_after:
    if nowlabel[0]==1:
        cv2.rectangle(image, (int(nowlabel[2]*px), int(nowlabel[3]*py)),(int(nowlabel[4]*px), int(nowlabel[5]*py)), 255, 5)
# rect = plt.Rectangle((nowlabel[1]*px-nowlabel[3]*px/2, nowlabel[2]*py-nowlabel[4]*py/2), nowlabel[3]*px, nowlabel[4]*py, fill=False, edgecolor = 'red',linewidth=1)
cv2.imshow('1',image)
# plt.imshow(image) # 图像数组
# plt.show()
