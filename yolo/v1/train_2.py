import os
import datetime
from copy import deepcopy
from operator import *
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import DataLoad
from net import MyNet
import torch
import time

from my_arguments import Args
from yolo.v1.util import FLOPs_and_Params
from yolo.v1.val import visualize
from yolo.v2.gen_anchor import get_anchor
from yolo.v2.yolov2 import YOLOv2

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes


class train(object):
    """
       网络训练接口，
       __train(): 训练过程函数
       __validate(): 验证过程函数
       __save_model(): 保存模型函数
       main(): 训练网络主函数
    """

    def __init__(self, opts):
        """
           :param opts: 命令行参数
        """
        self.opts = opts
        print("=======================Start training.=======================")

    @staticmethod
    def __save_model(model, epoch, opts):
        """
        保存第epoch个网络的参数
        :param model: torch.nn.Module, 需要训练的网络
        :param epoch: int, 表明当前训练的是第几个epoch
        :param opts: 命令行参数
        """
        model_name = "epoch%d.pth" % epoch
        save_dir = os.path.join(opts.checkpoints_dir, model_name)
        torch.save(model, save_dir)

    def main(self, model_type):
        """
        训练接口主函数，完成整个训练流程
        1. 创建训练集和验证集的DataLoader类
        2. 初始化带训练的网络
        3. 选择合适的优化器
        4. 训练并验证指定个epoch，保存其中评价指标最好的模型，并打印训练过程信息
        """
        opts = self.opts
        if not os.path.exists(opts.checkpoints_dir):
            os.mkdir(opts.checkpoints_dir)
        # random_seed = opts.random_seed
        data_dir='D:/pycharm/yolov1/dataset'
        train_dataset = DataLoad.trainData(model_type,data_dir)

        train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
        num_train = len(train_dataset)

        if opts.pretrain is None:
            if model_type == "v1":
                model = MyNet(opts)
            elif model_type == "v2":
                model = YOLOv2(opts)
                anchor=get_anchor(data_dir+'/trainlabel',opts)
                anchor=anchor.to(opts.GPU_id)
        else:
            model = torch.load(opts.pretrain)

        model.float()
        model.train()

        # if opts.use_GPU:
        model.to(opts.GPU_id)
        # 使用 tensorboard 可视化训练过程
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter("logs")

        # 构建训练优化器
        # optimizer = torch.optim.SGD(model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for e in range(opts.start_epoch, opts.epoch + 1000):
            t = time.time()
            device = opts.GPU_id
            avg_loss = 0.  # 平均损失数值

            # log_file是保存网络训练过程信息的文件，网络训练信息会以追加的形式打印在log.txt里，不会覆盖原有log文件
            log_file = open(os.path.join(opts.checkpoints_dir, "log.txt"), "a+")
            localtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 打印训练时间
            log_file.write(localtime)
            log_file.write("\n======================training epoch %d======================\n" % e)

            for i, (imgs, labels) in enumerate(train_loader):
                # if opts.use_GPU:
                imgs = imgs.to(device)
                labels = labels.to(device)
                # writer.add_images('img',imgs,e*100+i)


                conf_loss, cls_loss, bbox_loss, total_loss = model(inputs=imgs, labels=labels,anchor=anchor,model_type=model_type)  # 前向传播+计算损失

                optimizer.zero_grad()  # 梯度清零
                total_loss.backward()  # 反向传播
                optimizer.step()  # 优化网络参数
                avg_loss = (avg_loss * i + total_loss.item()) / (i + 1)
                if i % 100 == 0:  # 根据打印频率输出log信息和训练信息
                    print("Epoch %d/%d | Iter %d/%d | training loss = %.3f, avg_loss = %.3f" %
                          (e, opts.epoch, i, num_train // opts.batch_size, total_loss.item(), avg_loss))
                    log_file.write("Epoch %d/%d | Iter %d/%d | training loss = %.3f, avg_loss = %.3f\n" %
                                   (e, opts.epoch, i, num_train // opts.batch_size, total_loss.item(), avg_loss))
                    log_file.flush()

                    # 查看训练结果
                    bboxes, scores, train_labels = model(inputs=imgs, labels=labels, trainable=True,anchor=anchor)
                    scale = np.array([[416, 416, 416, 416]])
                    bboxes = bboxes*scale

                    class_colors = [(np.random.randint(255),
                                     np.random.randint(255),
                                     np.random.randint(255)) for _ in range(opts.num_classes)]
                    # 可视化检测结果
                    img_processed = visualize(
                        img=imgs[0, ...].squeeze().detach().permute(1, 2, 0).clamp(0, 255).cpu().numpy().astype(int),
                        bboxes=bboxes,
                        scores=scores,
                        labels=train_labels,
                        vis_thresh=0.3,
                        class_colors=class_colors,
                        # class_names=list(range(80)),
                        class_names=list(range(opts.num_classes)),
                        class_indexs=None,
                    )
                    if e % 100 == 0:
                        # plt.savefig("./%s.jpg" % e)
                        # writer.add_images("result_img_box",torch.tensor(img_processed[None,...]).permute(0,3, 1, 2),e*10+i)
                        # cv2.imshow('1',np.array(img_processed, dtype=np.uint8))
                        plt.imshow(img_processed)
                        plt.show()
                        cv2.waitKey(2)

            log_file.close()

            t2 = time.time()
            # writer.add_scalar("loss", total_loss, e)
            print("Training consumes %.2f second\n" % (t2 - t))
            with open(os.path.join(opts.checkpoints_dir, "log.txt"), "a+") as log_file:
                log_file.write("Training consumes %.2f second\n" % (t2 - t))
        writer.close()


if __name__ == '__main__':
    # 训练网络代码
    args = Args()
    args.set_train_args()  # 获取命令行参数
    train_interface = train(args.get_opts())
    train_interface.main(model_type="v2")  # 调用训练接口
