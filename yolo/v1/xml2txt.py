import xml.etree.ElementTree as ET
import os
from os import getcwd
from os.path import join
import glob

sets = ['train', 'test']  # 分别保存训练集和测试集的文件夹名称
classes = ['0',
           '1','2', '3', '4', '5', '6',
           '7','8','9','10','11','12','13',
           '14','15','16','17','18','19','20']  # 标注时的标签
class_name=['person',
            'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
'''
xml中框的左上角坐标和右下角坐标(x1,y1,x2,y2)
》》txt中的中心点坐标和宽和高(x,y,w,h)，并且归一化
'''


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(data_dir, image_id):
    in_file = open(data_dir + '/annotations/%s.xml' % image_id)  # 读取xml
    out_file = open(data_dir + '/labels/%s.txt' % image_id, 'w')  # 保存txt

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text) # 图片宽
    h = int(size.find('height').text) # 图片高
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in class_name:
            print('无当前标签：',cls)
            return
        elif int(difficult) == 1:
            continue
        cls_id = class_name.index(cls)  # 获取类别索引
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str('%.6f' % a) for a in bb]) + '\n')


wd = getcwd()
print(wd)  # 当前路径

data_dir = r'D:\pycharm\datasets\voc2007'

image_ids = []
for x in glob.glob(data_dir +'/Annotations'+ '/*.xml'):
    image_ids.append(os.path.basename(x)[:-4])
print('数量:', len(image_ids))  # 确认数量
i = 0
for image_id in image_ids:
    i = i + 1
    convert_annotation(data_dir, image_id)

print("Done!!!")
