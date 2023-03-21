import numpy as np
import cv2
import torch
from thop import profile
from torch import Tensor, nn


# CBL模块
class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, padding=0, d=1, g=1, act=True):
        super(Conv, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(c1, c2, k, stride=s, padding=padding, dilation=d, groups=g),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()
        )

    def forward(self, x):
        return self.convs(x)


def calculate_iou(bbox1, bbox2):
    """计算bbox1=(x1,y1,x2,y2)和bbox2=(x3,y3,x4,y4)两个bbox的iou"""
    if bbox1[2] <= bbox1[0] or bbox1[3] <= bbox1[1] or bbox2[2] <= bbox2[0] or bbox2[3] <= bbox2[1]:
        return 0  # 如果bbox1或bbox2没有面积，或者输入错误，直接返回0

    intersect_bbox = [0., 0., 0., 0.]  # bbox1和bbox2的重合区域的(x1,y1,x2,y2)

    intersect_bbox[0] = max(bbox1[0], bbox2[0])
    intersect_bbox[1] = max(bbox1[1], bbox2[1])
    intersect_bbox[2] = min(bbox1[2], bbox2[2])
    intersect_bbox[3] = min(bbox1[3], bbox2[3])

    w = max(intersect_bbox[2] - intersect_bbox[0], 0)
    h = max(intersect_bbox[3] - intersect_bbox[1], 0)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])  # bbox1面积
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])  # bbox2面积
    area_intersect = w * h  # 交集面积
    iou = area_intersect / (area1 + area2 - area_intersect + 1e-6)  # 防止除0
    # print(bbox1,bbox2)
    # print(intersect_bbox)
    # input()
    return iou


# YOLOv5 IOU
def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)  # 每个框的面积 (N,)
    area2 = box_area(boxes2)  # (M,)

    lt = torch.max(boxes1[:, :, :2], boxes2[:, :, :2])  # 选择左上角最大的xy，即相交矩形左上角
    rb = torch.min(boxes1[:, :, 2:], boxes2[:, :, 2:])  # 选择有下角最小xy，即相交矩形右下角

    wh = (rb - lt).clamp(min=0)  # 小于0的为0  clamp 钳；夹钳；
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou  # NxM， boxes1中对应框与boxes2中对应位置框的IOU；


# YOLOv2 IOU
def yolov2_box_iou(boxes1: Tensor, boxes2: Tensor, just_wh=False, is_iteration=False, boxes2_expand=None) -> Tensor:
    # boxes1: [bsize, h * w, 锚框数, 4]
    if just_wh:
        boxes1[..., :2] = 0
        boxes2[..., :2] = 0
    area1 = box_area(boxes1)  # 每个框的面积 (N,)
    area2 = box_area(boxes2)  # (M,)

    lt = torch.max(boxes1[..., :2], boxes2[..., :2])  # 选择左上角最大的xy，即相交矩形左上角
    rb = torch.min(boxes1[..., 2:], boxes2[..., 2:])  # 选择有下角最小xy，即相交矩形右下角

    wh = (rb - lt).clamp(min=0)  # 小于0的为0  clamp 钳；夹钳；
    inter = wh[..., 0] * wh[..., 1]  # [N,M]

    iou = inter / (area1 + area2 - inter)

    if is_iteration:
        an_num = boxes1.size(2)  # an框数量，5
        sum_gt_obj = torch.max(boxes2_expand.sum(1))
        out_box = torch.tensor(np.zeros((boxes1.size(0), boxes1.size(1), an_num, int(sum_gt_obj))))  # [bs, h*w, an数, gt框数]
        now_boxes2 = boxes2.view(boxes1.size(0), boxes1.size(1), 4)
        for i in range(boxes1.size(0)):  # bs循环
            where_obj = torch.where(boxes2_expand[i, :] > 0)
            for j in range(len(where_obj[0])):  # gt框数量循环
                mid_where = where_obj[0][j]
                out_box[i, :, :, j] = yolov2_box_iou(boxes1[i, ...], now_boxes2[i, mid_where, :][None, None, :])
        return out_box

    return iou  # NxM， boxes1中对应框与boxes2中对应位置框的IOU；


# 计算box的面积
def box_area(boxes: Tensor) -> Tensor:
    """
    [Batch,H*W,(x,y,w,h)]
    x,y为相对百分比位置
    w,h为绝对百分比
    """
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])


def yolov2_box_area(boxes: Tensor) -> Tensor:
    # [Batch, H * W, anchor_num, (x, y, w, h)]
    return (boxes[:, :, :, 2] - boxes[:, :, :, 0]) * (boxes[:, :, :, 3] - boxes[:, :, :, 1])


# 解算边界框
def decode_boxes(pred, opts, model_type='v1',anchor=None):
    """
        将txtytwth转换为常用的x1y1x2y2形式。
    """
    matsize=opts.output_mat_size
    output = torch.zeros_like(pred)
    if model_type == 'v1':
        pred[..., :2] = (pred[..., :2] * matsize + create_grid(matsize).to(opts.GPU_id)) / matsize  # 加上坐标
        # 将所有bbox的中心带入坐标和宽高换算成x1y1x2y2形式
        output[..., :2] = pred[..., :2] - pred[..., 2:] * 0.5
        output[..., 2:] = pred[..., :2] + pred[..., 2:] * 0.5
    elif model_type == 'v2':
        pred[..., :2]=(pred[..., :2]+ create_grid(matsize).to(opts.GPU_id))/ matsize
        output[..., :2] = pred[..., :2] - pred[..., 2:] * 0.5
        output[..., 2:] = pred[..., :2] + pred[..., 2:] * 0.5
    return output

def get_maxanchor(pred,anchor):
    new_out_pred=torch.zeros(pred.size(0),pred.size(1),pred.size(3))
    new_out_anchor=torch.zeros(pred.size(0),pred.size(1),anchor.size(1))
    for i in range(pred.size(0)):
        for j in range(pred.size(1)):
            an_index=torch.max(pred[i,j,:,4],-1).indices
            new_out_pred[i,j,:]=pred[i,j,an_index,:]
            new_out_anchor[i,j,:]=anchor[an_index,:]
    return new_out_pred,new_out_anchor

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


# 计算浮点数和参数数量
def FLOPs_and_Params(model, img_size, device):
    x = torch.randn(1, 3, img_size, img_size).to(device)
    print('==============================')
    flops, params = profile(model, inputs=(x,))
    print('==============================')
    print('FLOPs : {:.2f} B'.format(flops / 1e9))
    print('Params : {:.2f} M'.format(params / 1e6))


# yolov5：图片放缩
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """用在LoadImagesAndLabels模块的__getitem__函数  只在val时才会使用
        将图片缩放调整到指定大小
        Resize and pad image while meeting stride-multiple constraints
        https://github.com/ultralytics/yolov3/issues/232
        img: 原图 hwc (形状是 (h,w,c)  高、宽、通道（RGB）  像素值范围是0-255 )
        new_shape: 缩放后的最长边大小
        color: pad的颜色
        auto: True 保证缩放后的图片保持原图的比例 即 将原图最长边缩放到指定大小，再将原图较短边按原图比例缩放（不会失真）
              False 将原图最长边缩放到指定大小，再将原图较短边按原图比例缩放,最后将较短边两边pad操作缩放到最长边大小（不会失真）
        scale_fill: True 简单粗暴的将原图resize到指定的大小 相当于就是resize 没有pad操作（失真）
        scale_up: True  对于小于new_shape的原图进行缩放,大于的不变
                  False 对于大于new_shape的原图进行缩放,小于的不变
        :return: img: letterbox后的图片 hwc
                 ratio: wh ratios
                 (dw, dh): w和h的pad
    """
    shape = im.shape[:2]  # current shape [height, width]  第一层resize后图片大小
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 只进行下采样 因为上采样会让图片模糊
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # 保证缩放后图像比例不变
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle 保证原图比例不变，将图像最大边缩放到指定大小
        # 这里的取余操作可以保证padding后的图片是32的整数倍(416x416)，如果是(512x512)可以保证是64的整数倍
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch 简单粗暴的将图片缩放到指定尺寸
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
    # 在较小边的两侧进行pad, 而不是在一侧pad
    dw /= 2  # divide padding into 2 sides 将padding分到上下，左右两侧  dw=0
    dh /= 2

    if shape[::-1] != new_unpad:  # resize 将原图resize到new_unpad（长边相同，比例相同的新图）
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    return im, ratio, (dw, dh)


# 加载权重
def load_weight(model, path_to_ckpt=None):
    # check
    if path_to_ckpt is None:
        print('no weight file ...')
        return model

    checkpoint_state_dict = torch.load(path_to_ckpt, map_location='cpu')
    # model state dict
    model_state_dict = model.state_dict()
    # check
    for k in list(checkpoint_state_dict.keys()):
        if k in model_state_dict:
            shape_model = tuple(model_state_dict[k].shape)
            shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                checkpoint_state_dict.pop(k)
        else:
            checkpoint_state_dict.pop(k)
            print(k)

    model.load_state_dict(checkpoint_state_dict)
    print('Finished loading model!')

    return model


# yolov2：CBL+passthrough
class ReorgLayer(nn.Module):
    def __init__(self, c1, c2, k, s=1, padding=0, d=1, g=1):
        super(ReorgLayer, self).__init__()
        self.conv = Conv(c1, c2, k, s, padding, d, g)

    def forward(self, input_x, opts):
        # input_x:[bsize, c, h, w]
        x = self.conv(input_x)
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


# yolov2：CONV+LeakyReLU
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, same_padding=False, LeakyReLU_size=0.1):
        super(Conv2d, self).__init__()

        # 计算padding
        padding = int((kernel_size - 1) / 2) if same_padding else 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding=padding)
        self.relu = nn.LeakyReLU(LeakyReLU_size, inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# 实现递归式新建网络层
def make_layers(in_channels, net_cfg):
    layers = []

    if len(net_cfg) > 0 and isinstance(net_cfg[0], list):
        for sub_cfg in net_cfg:
            layer, in_channels = make_layers(in_channels, sub_cfg)
            layers.append(layer)
    else:
        for item in net_cfg:
            if item == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_channels, ksize = item
                layers.append(Conv(in_channels,
                                   out_channels,
                                   ksize))
                # layers.append(net_utils.Conv2d(in_channels, out_channels,
                #     ksize, same_padding=True))
                in_channels = out_channels

    return nn.Sequential(*layers), in_channels


# GIOU ，没试过
def GIOU(self, box1, box2):
    box1x1 = box1[:, 0]  # xmin
    box1y1 = box1[:, 1]  # ymin
    box1x2 = box1[:, 2]  # xmax
    box1y2 = box1[:, 3]  # ymax

    box2x1 = box2[:, 0]  # xmin
    box2y1 = box2[:, 1]  # ymin
    box2x2 = box2[:, 2]  # xmax
    box2y2 = box2[:, 3]  # ymax

    IOU = (np.minimum(box2x2, box1x2) - np.maximum(box1x1, box2x1)) \
          * \
          (np.minimum(box2y2, box1y2) - np.maximum(box2y1, box1y1)) / (
                  (box1x2 - box1x1) * (box1y2 - box1y1) + (box2x2 - box2x1) * (box2y2 - box2y1) - (
                  np.minimum(box2x2, box1x2) - np.maximum(box1x1, box2x1))
          )
    # C,最小外接矩阵
    C = (np.maximum(box2x2, box1x2) - np.minimum(box1x1, box2x1)) * (
            np.maximum(box2x2, box1x2) - np.minimum(box1x1, box2x1))

    GIOU = IOU - \
           np.abs(C - (box1x2 - box1x1) * (box1y2 - box1y1) + (box2x2 - box2x1) * (box2y2 - box2y1) - (
                   np.minimum(box2x2, box1x2) - np.maximum(box1x1, box2x1))) / np.abs(C)

    return GIOU
