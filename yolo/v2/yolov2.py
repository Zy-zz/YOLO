import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from yolo.v1.util import ReorgLayer, Conv2d, make_layers, yolov2_box_iou, Conv, decode_boxes, create_grid, get_maxanchor


class YOLOv2(nn.Module):
    def __init__(self, opts):
        """
           :param opts: 命令行参数
        """
        self.global_average_pool = None
        self.anchor = None
        self.opts = opts
        super(YOLOv2, self).__init__()

        self.Layers1 = nn.Sequential(
            Conv(3, 32, k=3, padding=1),
            nn.MaxPool2d(2, 2),
            Conv(32, 64, k=3, padding=1),
            nn.MaxPool2d(2, 2),
            Conv(64, 128, k=3, padding=1),
            Conv(128, 64, k=1),
            Conv(64, 128, k=3, padding=1),
            nn.MaxPool2d(2, 2),
            Conv(128, 256, k=3, padding=1),
            Conv(256, 128, k=1),
            Conv(128, 256, k=3, padding=1),
            nn.MaxPool2d(2, 2),
            Conv(256, 512, k=3, padding=1),
            Conv(512, 256, k=1),
            Conv(256, 512, k=3, padding=1),
            Conv(512, 256, k=1),
            Conv(256, 512, k=3, padding=1),
        )

        self.Layers2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            Conv(512, 1024, k=3, padding=1),
            Conv(1024, 512, k=1),
            Conv(512, 1024, k=3, padding=1),
            Conv(1024, 512, k=1),
            Conv(512, 1024, k=3, padding=1),
        )

        self.Layers3 = nn.Sequential(
            Conv(1024, 1024, k=3, padding=1),
            Conv(1024, 1024, k=3, padding=1),
        )

        self.Layers4 = nn.Sequential(
            Conv(1280, 1024, k=3, padding=1),
            Conv(1024, self.opts.anchor_num * (self.opts.num_classes + 5), k=1),
        )

        self.reorg = ReorgLayer(512, 64, k=1)

        self.global_average_pool_layer = nn.AvgPool2d((1, 1))  # 平均池化

        # train
        self.bbox_loss = None
        self.iou_loss = None
        self.cls_loss = None

    def forward(self, inputs, labels, anchor=None, trainable=False,model_type='v1'):
        self.anchor=anchor
        conv1s = self.Layers1(inputs)
        conv2 = self.Layers2(conv1s)
        conv3 = self.Layers3(conv2)
        # bsize, c, h, w
        conv1s_reorg = self.reorg(conv1s, self.opts)
        cat_1_3 = torch.cat([conv1s_reorg, conv3], 1)
        conv4 = self.Layers4(cat_1_3)
        self.global_average_pool = self.global_average_pool_layer(conv4)

        bsize, _, h, w = self.global_average_pool.size()

        # 判断是否测试
        if trainable:
            return self.inference()

        # [bsize, c, h, w] -> [bsize, h, w, c] -> [bsize, h * w, 锚框数, 5+类别数]
        global_average_pool_reshaped = self.global_average_pool.permute(0, 2, 3, 1).contiguous().view(bsize, -1, self.opts.anchor_num, self.opts.num_classes + 5)

        # [tx, ty, tw, th, to] -> [sig(tx), sig(ty), exp(tw), exp(th), sig(to)]
        xy_pred = F.sigmoid(global_average_pool_reshaped[:, :, :, 0:2])
        wh_pred = torch.exp(global_average_pool_reshaped[:, :, :, 2:4])
        bbox_pred = torch.cat([xy_pred, wh_pred], 3)  # 预测框的中心点和宽高拼接起来
        conf_pred = F.sigmoid(global_average_pool_reshaped[:, :, :, 4:5])  # 置信度

        class_pred = global_average_pool_reshaped[:, :, :, 5:].contiguous()  # 类别
        # prob_class_pred = nn.Softmax(class_pred.view(-1, class_pred.size()[-1])).dim.view_as(class_pred)  # 类别取归一化

        (
            conf_loss,
            cls_loss,
            bbox_loss,
            total_loss
        ) = compute_loss(pred_conf=conf_pred,
                         pred_cls=class_pred,
                         pred_txtytwth=bbox_pred,
                         targets=labels,
                         opts=self.opts,
                         anchor=anchor
                         )

        return conf_loss, cls_loss, bbox_loss, total_loss

    @torch.no_grad()
    def inference(self):
        bsize, _, h, w = self.global_average_pool.size()
        # [bsize, c, h, w] -> [bsize, h, w, c] -> [bsize, h * w, 锚框数, 5+类别数]
        global_average_pool_reshaped = self.global_average_pool.permute(0, 2, 3, 1).contiguous().view(bsize, -1, self.opts.anchor_num, self.opts.num_classes + 5)

        # global_average_pool_reshaped:[bs,h*w,5+类别数]
        # new_anchor:[bs,h*w,4]
        global_average_pool_reshaped,new_anchor=get_maxanchor(global_average_pool_reshaped,self.anchor)

        # [tx, ty, tw, th, to] -> [sig(tx), sig(ty), exp(tw), exp(th), sig(to)]
        xy_pred = torch.sigmoid(global_average_pool_reshaped[..., 0:2])
        wh_pred = torch.exp(global_average_pool_reshaped[..., 2:4])
        bbox_pred=torch.cat([xy_pred, wh_pred], -1).to(self.opts.GPU_id)
        bbox_pred[...,2:] = bbox_pred[...,2:]* new_anchor[...,2:].to(self.opts.GPU_id) # 预测框的中心点和宽高拼接起来
        conf_pred = torch.sigmoid(global_average_pool_reshaped[..., 4:5])  # 置信度,[bs,h*w,1]
        class_pred = global_average_pool_reshaped[..., 5:].contiguous()  # 类别
        # prob_class_pred = F.softmax(class_pred.view(-1, class_pred.size()[-1])).view_as(class_pred)  # 类别取归一化

        # 测试时batchsize是1，用[0]将其取走。
        conf_pred = conf_pred[0]  # [H*W, 1]
        cls_pred = class_pred[0]  # [H*W, NC]
        txtytwth_pred = bbox_pred[0]  # [H*W, 4]

        # 得分
        scores = torch.sigmoid(conf_pred) *cls_pred

        # 解算边界框, 并归一化边界框: [H*W, 4]
        bboxes = decode_boxes(txtytwth_pred,opts=self.opts,model_type='v2')
        bboxes = torch.clamp(bboxes, 0., 1.)

        # 将预测放在cpu处理上，以便进行后处理
        scores = scores.to('cpu').detach().numpy()
        bboxes = bboxes.to('cpu').detach().numpy()

        # 后处理
        bboxes, scores, labels = self.postprocess(bboxes, scores)

        return bboxes, scores, labels

    def postprocess(self,bboxes, scores):
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


def compute_loss(pred_conf, pred_cls, pred_txtytwth, targets, opts, anchor):
    """
    :param pred_conf: [bsize, h * w, 锚框数] ，置信度
    :param pred_cls: [bsize, h * w, 锚框数, 类别数]，归一化后类别
    :param pred_txtytwth: [bsize, h * w, 锚框数, 4] x,y,w,h
    :param targets: [bsize, h * w, 6]，gt，类别+置信度+x,y,w,h
    :param anchor : [5, (x,y,w,h)] 先验框，5个
    :return:

    预测与gt + 先验框与预测 + gt与预测
    """
    batch_size = pred_conf.size(0)
    hw_num = pred_conf.size(1)
    anchor_num = pred_conf.size(2)  # 5

    # 损失函数
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    twth_loss_function = nn.MSELoss(reduction='none')

    # 标签
    gt_obj = targets[:, :, 0]  # [B, HW,] 置信度 ,默认1
    gt_cls = targets[:, :, 1].long()  # [B, HW,]
    gt_txty = targets[:, :, 2:4]  # [B, HW, 2]
    gt_twth = targets[:, :, 4:6]  # [B, HW, 2]

    # TODO:关于gt和所有an的IOU计算，是否可以提前先做了？gt是确定的，an也是确定的
    # [bs, h*w, an_num] ， 所有gt与所有cell下的5个an的IOU
    gt_anchor_iou = yolov2_box_iou(torch.concat((gt_txty, gt_twth), 2)[:, :, None, :], anchor[None, None, :, :], just_wh=True).view(batch_size, hw_num,
                                                                                                                                    anchor_num)
    # gt_anchor_iou_index_mid = gt_anchor_iou.sum(2).view(batch_size, hw_num).clone()  # 第三维取最大，得到[bs,h*w]，存储每个cell内an与gt的IOU的sum
    # gt_anchor_iou_index = torch.tensor(np.zeros(batch_size, hw_num))
    # gt_anchor_iou_index[torch.where(gt_anchor_iou_index_mid > 0)] += 1  # 用于判断每个cell内是否有gt，因为有gt一定有IOU
    gt_anchor_iou_max = torch.max(gt_anchor_iou, 2)  # 用于IOU值 ，记录每个cell内最大IOU，值和位置，位置信息就表示了每个cell里和gt匹配的是哪个an

    # 预测与gt
    # [bs, h*w, anchor_num, sum(gt_obj)]
    # pred_txtytwth: [bsize, h * w, 锚框数, 4]
    # targets: [bsize, h * w, 6]
    bbox_iou = yolov2_box_iou(pred_txtytwth, targets[:, :, None, 2:6], is_iteration=True, boxes2_expand=gt_obj)

    # IOU计算,[bs, h*w, pred的anchor_num, sum(gt_obj)]，存放所有bs，所有点，预测的5个锚框和label所有框的IOU
    bbox_iou[bbox_iou >= opts.Max_iou] = 1  # 让大于0.6的部分为1
    bbox_iou[bbox_iou < opts.Max_iou] = 0  # 让小于0.6的部分为0

    # 背景误差
    over_thresh_index = torch.where(bbox_iou.sum(3) > 0)  # 存放所有cell中，5个an各自是否超过0.6的阈值
    bbox_iou_2d = torch.tensor(np.zeros((batch_size, hw_num)))  # 用来存放这个bs下这个点是否有满足条件预测框，用于后续计算有目标的置信度
    bbox_iou_2d[over_thresh_index[0], over_thresh_index[1]] = bbox_iou_2d[over_thresh_index[0], over_thresh_index[1]]+1  # 5个预测框和所有的gt做IOU，只要有一个IOU大于0.6就不算背景

    # [bs, h*w, 锚数:5]
    bbox_iou_3d = torch.tensor(np.zeros((batch_size, hw_num, anchor_num)))  # 用于计算后续预测框的置信度
    bbox_iou_3d[over_thresh_index[0], over_thresh_index[1], over_thresh_index[2]] = bbox_iou_3d[over_thresh_index[0], over_thresh_index[1], over_thresh_index[2]]+1  # 存放5个预测框满足条件的框

    # 无目标置信度误差
    BG_loss = (1 - bbox_iou_3d.to(0)) * torch.square(pred_conf.view_as(bbox_iou_3d))  # (1-bbox_iou):留下IOU没超过阈值的部分

    # 有目标损失计算准备
    noan_3d_pred_cls = torch.tensor(np.zeros((batch_size, hw_num, opts.num_classes))).to(0)  # [bs,h*w,类别数]
    anchor_xywh_3D = torch.tensor(np.zeros((batch_size, hw_num, 4))).to(0)  # [bs,h*w,(xywh)] anchor的xywh
    pred_xywh_3D = torch.tensor(np.zeros((batch_size, hw_num, 4))).to(0)  # [bs,h*w,(xywh)] 预测框的xywh
    pred_conf_2d = torch.tensor(np.zeros((batch_size, hw_num))).to(0)  # [bs,h*w] 预测框的置信度
    for i in range(pred_cls.size(0)):  # 循环bs
        for j in range(pred_cls.size(1)):  # 循环h*w
            if gt_obj[i, j] != 0:
                which_anchor = gt_anchor_iou_max.indices[i, j]  # 中心点所落的cell里与之对应的an的索引
                anchor_xywh_3D[i, j, :] = anchor[which_anchor, :]
            else:
                continue
            if bbox_iou_2d[i, j] == 0:  # 该cell无gt跳过，该cell的预测若和所有gt的IOU都不超过阈值则跳过
                continue
            which_anchor = gt_anchor_iou_max.indices[i, j]
            noan_3d_pred_cls[i, j, :] = pred_cls[i, j, which_anchor, :]
            pred_xywh_3D[i, j, :] = pred_txtytwth[i, j, which_anchor, :]
            pred_xywh_3D[i, j, 2:] = pred_xywh_3D[i, j, 2:]*anchor_xywh_3D[i, j, 2:]
            pred_conf_2d[i, j] = pred_conf[i, j, which_anchor]

    # 有目标置信度误差
    obj_loss = (gt_obj * torch.square(gt_anchor_iou_max.values - pred_conf_2d.to(0)))

    # 将gt的类别独热
    one_hot_gt_cls = torch.nn.functional.one_hot(gt_cls.long(), opts.num_classes)

    # 类别损失
    cls_loss = 1 * (gt_obj[...,None] *torch.square(one_hot_gt_cls - noan_3d_pred_cls.long().to(0))).sum()

    # anchor与预测框坐标误差
    pred_xywh_3D[..., 2:]=pred_xywh_3D[..., 2:]*anchor_xywh_3D[..., 2:]
    pred_xy_3D=pred_xywh_3D[..., :2].clone()
    pred_wh_3D=pred_xywh_3D[..., 2:].clone()
    anchor_xy_3D=anchor_xywh_3D[..., :2].clone()
    anchor_wh_3D = anchor_xywh_3D[..., 2:].clone()
    anchor_pred_xy_loss = torch.square(anchor_xy_3D- pred_xy_3D)
    anchor_pred_wh_loss = torch.square(anchor_wh_3D- pred_wh_3D)

    # 与gt匹配的anchor坐标误差
    gt_anchor_xy_loss = torch.square(gt_txty- anchor_xywh_3D[..., :2])
    gt_anchor_wh_loss = torch.square(gt_twth- anchor_xywh_3D[..., 2:])

    # 总置信度损失，有目标+无目标（背景）
    conf_loss = 5 * obj_loss.sum() + 1 * BG_loss.sum()

    # 总bbox损失：gt与匹配的anchor，匹配的an与pred
    an_pred_bbox_loss = 1 * anchor_pred_xy_loss.sum() + 1 * anchor_pred_wh_loss.sum()
    gt_an_bbox_loss = 1 * gt_anchor_xy_loss.sum() + 1 * gt_anchor_wh_loss.sum()
    bbox_loss = an_pred_bbox_loss + gt_an_bbox_loss

    total_loss = conf_loss + cls_loss + bbox_loss
    return conf_loss, cls_loss, bbox_loss, total_loss
