import math

import numpy as np
import torch
import torch.nn as nn

from yolo.v1.util import calculate_iou, box_iou, decode_boxes
import cv2
import matplotlib.pyplot as plt
from yolo.v1.util import Conv


class MyNet(nn.Module):

    def __init__(self, opts):
        super(MyNet, self).__init__()
        self.opts = opts
        self.pred = None
        # 进入的通道数，出的通道数，7*7卷积，步长，padding
        self.L1 = Conv(3, 64, k=7, s=2, padding=3)

        # 2*2窗口大小，步长
        self.MaxPool1 = nn.MaxPool2d(2, 2)

        self.L2 = Conv(64, 192, k=3, padding=1)
        self.MaxPool2 = nn.MaxPool2d(2, 2)
        self.Layers1 = nn.Sequential(
            Conv(192, 128, k=1),
            Conv(128, 256, k=3, padding=1),
            Conv(256, 256, 1),
            Conv(256, 512, k=3, padding=1),
        )
        self.MaxPool3 = nn.MaxPool2d(2, 2)
        self.Layers2 = nn.Sequential(
            Conv(512, 256, 1),
            Conv(256, 512, 3, padding=1),
            Conv(512, 256, 1),
            Conv(256, 512, 3, padding=1),
            Conv(512, 256, 1),
            Conv(256, 512, 3, padding=1),
            Conv(512, 256, 1),
            Conv(256, 512, 3, padding=1),
            Conv(512, 512, 1),
            Conv(512, 1024, k=3, padding=1),
        )
        self.MaxPool4 = nn.MaxPool2d(2, 2)
        self.Layers3 = nn.Sequential(
                Conv(1024, 512, 1),
                Conv(512, 1024, 3, padding=1),
                Conv(1024, 512, 1),
                Conv(512, 1024, 3, padding=1),
                Conv(1024, 1024, 3, padding=1),
                Conv(1024, 1024, k=3, s=2, padding=1),
        )
        self.L3 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.L4 = nn.Conv2d(1024, 1024, 3, padding=1)
        # 以下是YOLOv1的最后2个全连接层
        self.Conn_layers = nn.Sequential(
            # Conv(1024,(5 + self.opts.num_classes),1),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.BatchNorm1d(4096),# BatchNorm1d 函数在 batchsize 为1时报错
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(4096, 7 * 7 * (5 + opts.num_classes)),
            nn.BatchNorm1d(7 * 7 * (5 + opts.num_classes)),
            nn.Sigmoid()  # 增加sigmoid函数是为了将输出全部映射到(0,1)之间，因为如果出现负数或太大的数，后续计算loss会很麻烦
        )
        # self.test = nn.Conv2d(1024, 90, 3, padding=1)
        # self.test2 = nn.Sigmoid()

    def forward(self, inputs, trainable=False, labels=None,cfg=None,anchor=None):
        x = self.L1(inputs)
        x = self.MaxPool1(x)
        x = self.L2(x)
        x = self.MaxPool2(x)
        x = self.Layers1(x)
        x = self.MaxPool3(x)
        x = self.Layers2(x)
        x = self.MaxPool4(x)
        x = self.Layers3(x)
        x = self.L3(x)
        x = self.L4(x)
        x = x.view(-1, 7 * 7 * 1024)
        x = self.Conn_layers(x)
        # self.pred=x
        self.pred = x.reshape(-1, (5 + self.opts.num_classes), 7, 7) # 记住最后要reshape一下输出数据
        # x = self.test(x)
        # self.pred = self.test2(x)

        # 判断是否测试
        if trainable:
            return self.inference()

        # 对pred 的size做一些view调整，便于后续的处理
        # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
        pred = self.pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

        # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测
        # [B, H*W, 1]
        conf_pred = pred[..., :1]
        # [B, H*W, num_cls]
        cls_pred = pred[..., 1:1 + self.opts.num_classes]
        # [B, H*W, 4]
        txtytwth_pred = pred[..., 1 + self.opts.num_classes:]

        # 计算损失
        (
            conf_loss,
            cls_loss,
            bbox_loss,
            total_loss
        ) = compute_loss(pred_conf=conf_pred,
                         pred_cls=cls_pred,
                         pred_txtytwth=txtytwth_pred,
                         targets=labels
                         )

        return conf_loss, cls_loss, bbox_loss, total_loss

    @torch.no_grad()
    def inference(self):
        # 对pred 的size做一些view调整，便于后续的处理
        # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
        pred = self.pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

        # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测
        # [B, H*W, 1]
        conf_pred = pred[..., :1]
        # [B, H*W, num_cls]
        cls_pred = pred[..., 1:1 + self.opts.num_classes]
        # [B, H*W, 4]
        txtytwth_pred = pred[..., 1 + self.opts.num_classes:]

        # 测试时batchsize是1，用[0]将其取走。
        conf_pred = conf_pred[0]  # [H*W, 1]
        cls_pred = cls_pred[0]  # [H*W, NC]
        txtytwth_pred = txtytwth_pred[0]  # [H*W, 4]

        # 双预测框时如下使用
        # new_txtytwth_pred = torch.tensor(np.zeros([49, 4]))
        # new_txtytwth_pred[:, 0] = torch.max(txtytwth_pred[:, [0, 2]], -1).values
        # new_txtytwth_pred[:, 1] = torch.max(txtytwth_pred[:, [1, 3]], -1).values
        # new_txtytwth_pred[:, 2] = torch.max(txtytwth_pred[:, [4, 6]], -1).values
        # new_txtytwth_pred[:, 3] = torch.max(txtytwth_pred[:, [5, 7]], -1).values
        # txtytwth_pred = new_txtytwth_pred
        # 每个边界框的得分
        # scores = torch.sigmoid(torch.max(conf_pred, -1).values)[..., None] * torch.softmax(cls_pred, dim=-1)

        scores = torch.sigmoid(conf_pred) * torch.softmax(cls_pred, dim=-1)

        # 解算边界框, 并归一化边界框: [H*W, 4]
        bboxes = decode_boxes(txtytwth_pred,opts=self.opts,model_type='v1')
        bboxes = torch.clamp(bboxes, 0., 1.)

        # 将预测放在cpu处理上，以便进行后处理
        scores = scores.to('cpu').detach().numpy()
        bboxes = bboxes.to('cpu').detach().numpy()

        # 后处理
        bboxes, scores, labels = self.postprocess(bboxes, scores)

        return bboxes, scores, labels



    def postprocess(self, bboxes, scores):
        """
        Input:
            bboxes: [HxW, 4]
            scores: [HxW, num_classes]
        Output:
            bboxes: [N, 4]
            score:  [N,]
            labels: [N,]
        """

        labels = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), labels)]

        # threshold
        keep = np.where(scores >= 0.01)  # conf_thresh 得分阈值，此处0.00001
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # NMS
        keep = np.zeros(len(bboxes), dtype=np.int)
        # 一次找一个类型，框不一定就一个
        # for i in range(80):
        for i in range(self.opts.num_classes):
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        return bboxes, scores, labels

    # 非极大值抑制
    def nms(self, bboxes, scores):
        """Pure Python NMS baseline."""
        x1 = bboxes[:, 0]  # xmin
        y1 = bboxes[:, 1]  # ymin
        x2 = bboxes[:, 2]  # xmax
        y2 = bboxes[:, 3]  # ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # 计算交集的左上角点和右下角点的坐标
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            # 计算交集的宽高
            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)
            # 计算交集的面积
            inter = w * h

            # 计算交并比
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # 滤除超过nms阈值的检测框
            inds = np.where(iou <= 0.5)[0]
            order = order[inds + 1]

        return keep


# 计算可信度误差
class MSEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(MSEWithLogitsLoss, self).__init__()

    def forward(self, logits, targets):
        inputs = torch.clamp(torch.sigmoid(logits), min=1e-14, max=1.0 - 1e-14)

        pos_id = (targets == 1.0).float()
        neg_id = (targets == 0.0).float()
        pos_loss = pos_id * (inputs - targets) ** 2
        neg_loss = neg_id * (inputs) ** 2
        loss = pos_loss + 0.5 * neg_loss

        return loss

# 计算总体误差
def compute_loss(pred_conf, pred_cls, pred_txtytwth, targets):
    batch_size = pred_conf.size(0)

    # 损失函数
    conf_loss_function = MSEWithLogitsLoss()
    cls_loss_function = nn.CrossEntropyLoss(reduction='none')
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    twth_loss_function = nn.MSELoss(reduction='none')

    # 网络输出图像
    pred_conf_1 = pred_conf[:, :, 0]  # [B, HW,] 第一个框置信度
    # pred_conf_2 = pred_conf[:, :, 1]  # [B, HW,] 第二个框置信度
    pred_cls = pred_cls.permute(0, 2, 1)  # [B, Class, HW]
    pred_txty_1 = pred_txtytwth[:, :, :2]  # [B, HW, 2] 第一个框中心点
    # pred_txty_2 = pred_txtytwth[:, :, 4:6]  # [B, HW, 2] 第二个框中心点
    pred_twth_1 = pred_txtytwth[:, :, 2:]  # [B, HW, 2] 第一个框宽高
    # pred_twth_2 = pred_txtytwth[:, :, 6:]  # [B, HW, 2] 第二个框宽高

    # 标签
    gt_obj = targets[:, :, 0]  # [B, HW,] 置信度 ,默认1
    gt_cls = targets[:, :, 1].long()  # [B, HW,]
    gt_txty = targets[:, :, 2:4]  # [B, HW, 2]
    gt_twth = targets[:, :, 4:6]  # [B, HW, 2]
    # gt_box_scale_weight = targets[:, :, 6]  # [B, HW,]

    # 选择iou大的bbox作为负责物体
    # bbox1 = torch.concat((pred_txty_1, pred_twth_1), dim=2)
    # bbox2 = torch.concat((pred_txty_2, pred_twth_2), dim=2)
    # bbox_gt = torch.concat((gt_txty, gt_twth), dim=2)
    # iou1 = box_iou(bbox1, bbox_gt)
    # iou2 = box_iou(bbox2, bbox_gt)
    # pred_txty = torch.where(iou1[..., None] >= iou2[..., None], pred_txty_1, pred_txty_2)
    # pred_twth = torch.where(iou1[..., None] >= iou2[..., None], pred_twth_1, pred_twth_2)
    # pred_conf = torch.where(iou1 >= iou2, pred_conf_1, pred_conf_2)

    # 置信度损失
    conf_loss = conf_loss_function(pred_conf_1, gt_obj)
    conf_loss = conf_loss.sum() / batch_size

    # 类别损失
    cls_loss = cls_loss_function(pred_cls, gt_cls) * gt_obj
    cls_loss = cls_loss.sum() / batch_size

    # 边界框txty的损失
    txty_loss = 5 * txty_loss_function(pred_txty_1, gt_txty).sum(-1) * gt_obj  # * gt_box_scale_weight
    txty_loss = txty_loss.sum() / batch_size

    # 边界框twth的损失
    twth_loss = 5 * twth_loss_function(torch.sqrt(pred_twth_1), torch.sqrt(gt_twth)).sum(-1) * gt_obj  # * gt_box_scale_weight
    twth_loss = twth_loss.sum() / batch_size
    bbox_loss = txty_loss + twth_loss

    # 总的损失
    total_loss = conf_loss + cls_loss + bbox_loss

    return 1 * conf_loss, 1 * cls_loss, 1 * bbox_loss, 1 * total_loss


if __name__ == '__main__':
    # 自定义输入张量，验证网络可以正常跑通，并计算loss，调试用
    x = torch.zeros(5, 3, 448, 448)
    net = MyNet()
    a = net(x)
    labels = torch.zeros(5, 30, 7, 7)
    loss = net.calculate_loss(labels)
    print(loss)
    print(a.shape)
