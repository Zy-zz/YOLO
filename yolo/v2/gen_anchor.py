import numpy as np
import os
from pathlib import Path
import glob

import torch


# 定义Box类，描述bounding box的坐标
class Box():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


# 计算两个box在某个轴上的重叠部分
# x1是box1的中心在该轴上的坐标
# len1是box1在该轴上的长度
# x2是box2的中心在该轴上的坐标
# len2是box2在该轴上的长度
# 返回值是该轴上重叠的长度
def overlap(x1, len1, x2, len2):
    len1_half = len1 / 2
    len2_half = len2 / 2

    left = max(x1 - len1_half, x2 - len2_half)
    right = min(x1 + len1_half, x2 + len2_half)

    return right - left


# 计算box a 和box b 的交集面积
# a和b都是Box类型实例
# 返回值area是box a 和box b 的交集面积
def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0

    area = w * h
    return area


# 计算 box a 和 box b 的并集面积
# a和b都是Box类型实例
# 返回值u是box a 和box b 的并集面积
def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u


# 计算 box a 和 box b 的 iou
# a和b都是Box类型实例
# 返回值是box a 和box b 的iou
def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)


# 使用k-means ++ 初始化 centroids，减少随机初始化的centroids对最终结果的影响
# boxes是所有bounding boxes的Box对象列表
# n_anchors是k-means的k值
# 返回值centroids 是初始化的n_anchors个centroid
def init_centroids(boxes, n_anchors):
    centroids = []
    boxes_num = len(boxes)

    centroid_index = np.random.choice(boxes_num, 1)  # 随机选择一个整数，作为初始的聚类中心，该数小于所有标签的box的数量
    centroids.append(boxes[centroid_index[0]])

    # 循环生成n个中心点
    for centroid_index in range(0, n_anchors - 1):

        sum_distance = 0
        distance_list = []
        cur_sum = 0

        # 对所有box循环
        for box in boxes:
            min_distance = 1
            for centroid_i, centroid in enumerate(centroids):  # 循环所有的中心点
                distance = (1 - box_iou(box, centroid))
                if distance < min_distance:  # 对每一个box取到离它最近的中心点的距离
                    min_distance = distance
            sum_distance =sum_distance+ min_distance
            distance_list.append(min_distance)

        distance_thresh = sum_distance * np.random.random()  # 由于取的随机数，因此越大的距离，越有可能作为下一个中心

        for i in range(0, boxes_num):
            cur_sum =cur_sum+ distance_list[i]  # 循环box数量，累加每个box离它最近的中心点的距离
            if cur_sum > distance_thresh:  # 当超过限度，那么该点就是中心
                centroids.append(boxes[i])
                break

    return centroids


# 进行 k-means 计算新的centroids
# boxes是所有bounding boxes的Box对象列表
# n_anchors是k-means的k值
# centroids是所有簇的中心
# 返回值new_centroids 是计算出的新簇中心
# 返回值groups是n_anchors个簇包含的boxes的列表
# 返回值loss是所有box距离所属的最近的centroid的距离的和
def do_kmeans(n_anchors, boxes, centroids):
    loss = 0
    groups = []
    new_centroids = []
    for i in range(n_anchors):  # 预生成n个中心所需的groups和坐标集合
        groups.append([])
        new_centroids.append(Box(0, 0, 0, 0))

    for box in boxes:  # 循环所有box
        min_distance = 1
        group_index = 0
        for centroid_index, centroid in enumerate(centroids):  # 循环所有的中心点
            distance = (1 - box_iou(box, centroid))  # 计算距离
            if distance < min_distance:  # 选择出距离该box最近的中心点
                min_distance = distance
                group_index = centroid_index
        groups[group_index].append(box)  # groups[group_index]就是第group_index簇包含该box对象
        loss =loss+ min_distance
        new_centroids[group_index].w =new_centroids[group_index].w+ box.w  # 累加属于这簇的点的w，后面除簇内点数，就是该簇新的中心w
        new_centroids[group_index].h =new_centroids[group_index].h+ box.h

    for i in range(n_anchors):
        new_centroids[i].w /= len(groups[i])
        new_centroids[i].h /= len(groups[i])

    return new_centroids, groups, loss


# 计算给定bounding boxes的n_anchors数量的centroids
# label_path是训练集列表文件地址
# n_anchors 是anchors的数量
# loss_convergence是允许的loss的最小变化值
# grid_size * grid_size 是栅格数量
# iterations_num是最大迭代次数
# plus = 1时启用k means ++ 初始化centroids
def compute_centroids(label_path, n_anchors, loss_convergence, grid_size, iterations_num, plus=True):
    f = []
    img_list_txt = os.path.join(label_path)  # 储存图片位置的列表
    for p in img_list_txt if isinstance(img_list_txt, list) else [img_list_txt]:
        p = Path(p)  # os-agnostic
        if p.is_dir():  # dir
            f =f+ glob.glob(str(p / '**' / '*.*'), recursive=True)

    # 筛选出标签
    label_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in ['txt'])
    plt_x = []
    plt_y = []
    boxes = []
    for label_i in label_files:
        with open(label_i, 'r') as f:
            mid_label = np.loadtxt(f)
            mid_label = mid_label.tolist()
            for kk in mid_label:
                if isinstance(kk, list):
                    plt_x.append(float(kk[3]))
                    plt_y.append(float(kk[4]))
                    boxes.append(Box(0, 0, float(kk[3]), float(kk[4])))
                else:
                    plt_x.append(float(mid_label[3]))
                    plt_y.append(float(mid_label[4]))
                    boxes.append(Box(0, 0, float(mid_label[3]), float(mid_label[4])))
                    break

    # 这里是中心点初始化的两种方法，但是感觉还是用第一种k-means++比较好
    if plus:
        centroids = init_centroids(boxes, n_anchors)
    else:
        centroid_indices = np.random.choice(len(boxes), n_anchors)
        centroids = []
        for centroid_index in centroid_indices:
            centroids.append(boxes[centroid_index])

    # iterate k-means
    centroids, groups, old_loss = do_kmeans(n_anchors, boxes, centroids)
    iterations = 1
    while (True):
        centroids, groups, loss = do_kmeans(n_anchors, boxes, centroids)
        iterations = iterations + 1
        if iterations % 100 == 0:
            print("loss = %f" % loss)
        # 判断loss是不是在期望的阈值内，或者说已经收敛了
        if abs(old_loss - loss) < loss_convergence or iterations > iterations_num:
            break
        old_loss = loss

        # for centroid in centroids:
        #     print(centroid.w * grid_size, centroid.h * grid_size)

    # print result
    # for centroid in centroids:
    #     print("loss = %f" % loss)
    #     print("k-means result：\n")
    #     print(centroid.w * grid_size, centroid.h * grid_size)

    return plt_x, plt_y, centroids


def get_anchor(label_path, opts):
    n_anchors = opts.anchor_num
    loss_convergence = 1e-6
    grid_size = 13
    iterations_num = 1000
    plus = 1
    plt_x, plt_y, centroids = compute_centroids(label_path, n_anchors, loss_convergence, grid_size, iterations_num, plus)

    out_anchor=torch.Tensor(np.zeros((n_anchors,4)))
    for i in range(n_anchors):
        out_anchor[i,0]=centroids[i].x
        out_anchor[i,1]=centroids[i].y
        out_anchor[i,2]=centroids[i].w
        out_anchor[i,3]=centroids[i].h

    return out_anchor
    # import matplotlib.pyplot as plt
    # plt.scatter(plt_x,plt_y)
    # for i in centroids:
    #     plt.scatter(i.w,i.h,c='red')
