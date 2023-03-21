import argparse
import torch
import numpy as np
import cv2
import os
import time

from yolo.v1 import DataLoad
from yolo.v1.DataLoad import testData
from yolo.v1.net import MyNet
from yolo.v1.util import load_weight, letterbox

parser = argparse.ArgumentParser(description='YOLO Detection')
parser.add_argument('-d', '--dataset', default='voc',
                    help='voc, coco-val.')
parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                    help='data root')
parser.add_argument('-size', '--input_size', default=416, type=int,
                    help='输入图像尺寸')

parser.add_argument('-v', '--version', default='yolo',
                    help='yolo')
parser.add_argument('--weight', default=None,
                    type=str, help='模型权重的路径')
parser.add_argument('--conf_thresh', default=0.1, type=float,
                    help='得分阈值')
parser.add_argument('--nms_thresh', default=0.50, type=float,
                    help='NMS 阈值')
parser.add_argument('-vs', '--visual_threshold', default=0.3, type=float,
                    help='用于可视化的阈值参数')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='use cuda.')
parser.add_argument('--save', action='store_true', default=False,
                    help='save vis results.')

args = parser.parse_args()


def plot_bbox_labels(img, bbox, label=None, cls_color=None, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    img=img.astype(np.uint8).copy()
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)

    if label is not None:
        # plot title bbox
        cv2.rectangle(img, (x1, y1 - t_size[1]), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
        # put the test on the title bbox
        cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img


def visualize(img,
              bboxes,
              scores,
              labels,
              vis_thresh,
              class_colors,
              class_names,
              class_indexs=None,
              dataset_name='voc'):
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        if scores[i] > 0.7:
            cls_id = int(labels[i])
            if dataset_name == 'coco-val':
                cls_color = class_colors[cls_id]
                cls_id = class_indexs[cls_id]
            else:
                cls_color = class_colors[cls_id]

            if len(class_names) > 1:
                mess = '%s: %.2f' % (class_names[cls_id], scores[i])
            else:
                cls_color = [255, 0, 0]
                mess = None
            cls_color=[255, 255, 255]
            img = plot_bbox_labels(img, bbox, mess, cls_color, text_scale=ts)

    return img


if __name__ == '__main__':
    # 是否使用cuda
    device = torch.device("cuda")

    # 输入图像的尺寸
    input_size = 448

    img_list, label_list = testData("D:\\pycharm\\datasets")

    # 用于可视化，给不同类别的边界框赋予不同的颜色，为了便于区分。
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(80)]

    # 构建模型
    model = MyNet()

    # 加载已训练好的模型权重
    model = load_weight(model, args.weight)
    model.to(device).eval()
    print('Finished loading model!')

    for i in range(len(img_list)):
        img = cv2.imread(img_list[i])
        img = letterbox(img, new_shape=(448, 448), auto=False, scaleFill=True)
        label = torch.tensor(label_list[i])
        label = DataLoad.convert_bbox2labels(label,7)
        imgs = imgs.to(device)
        labels = label.to(device)

        t0 = time.time()
        # 前向推理
        preds = model(imgs)  # 前向传播
        bboxes, scores, labels = model.calculate_loss(trainable=True)
        print("detection time used ", time.time() - t0, "s")

        # 将预测的输出映射到原图的尺寸上去
        scale = np.array([[448, 448, 448, 448]])
        bboxes *= scale

        # 可视化检测结果
        img_processed = visualize(
            img=img,
            bboxes=bboxes,
            scores=scores,
            labels=labels,
            vis_thresh=args.visual_threshold,
            class_colors=class_colors,
            class_names=None,
            class_indexs=None,
            )
        cv2.imshow('detection', img_processed)
        cv2.waitKey(0)

        # 保存可视化结果
        if args.save:
            cv2.imwrite(os.path.join("yolov1\\result", str(i).zfill(6) + '.jpg'), img_processed)
