import argparse
import torch


class Args(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def set_train_args(self):
        """options for train"""
        self.parser.add_argument("--batch_size", type=int, default=2)
        self.parser.add_argument("--lr", type=float, default=0.000001, help="learning rate")
        self.parser.add_argument("--weight_decay", type=float, default=1e-4)
        self.parser.add_argument("--epoch", type=int, default=200, help="number of end epoch")
        self.parser.add_argument("--start_epoch", type=int, default=1, help="number of start epoch")
        self.parser.add_argument("--use_GPU", action="store_true", help="identify whether to use gpu")
        self.parser.add_argument("--GPU_id", type=int, default=None, help="device id")
        self.parser.add_argument("--dataset_dir", type=str, default=r"D:\pycharm\yolov1\dataset")
        self.parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
        self.parser.add_argument("--print_freq", type=int, default=20,
                                 help="print training information frequency (per n iteration)")
        self.parser.add_argument("--save_freq", type=int, default=1, help="save model frequency (per n epoch)")
        self.parser.add_argument("--num_workers", type=int, default=0, help="use n threads to read data")
        self.parser.add_argument("--pretrain", type=str,
                                 default=None,
                                 help="pretrain model path")
        self.parser.add_argument("--random_seed", type=int, default=0, help="random seed for split dataset")
        self.parser.add_argument("--Max_iou", type=float, default=0.6, help="yolov2中误差中的IOU阈值")
        self.parser.add_argument("--anchor_num", type=int, default=1, help="yolov2中锚框个数")
        self.parser.add_argument("--num_classes", type=int, default=1, help="class数量")
        self.parser.add_argument("--output_mat_size", type=int, default=13, help="输出矩阵的size")

        self.opts = self.parser.parse_args()

        if torch.cuda.is_available():
            self.opts.use_GPU = True
            self.opts.GPU_id = torch.cuda.current_device()
            print("use GPU %d to train." % (self.opts.GPU_id))
        else:
            print("use CPU to train.")

    def set_test_args(self):
        """options for inference"""
        self.parser.add_argument("--batch_size", type=int, default=1)
        self.parser.add_argument("--use_GPU", action="store_true", help="identify whether to use gpu")
        self.parser.add_argument("--GPU_id", type=int, default=None, help="device id")
        self.parser.add_argument("--dataset_dir", type=str,
                                 default=r"D:\pycharm\yolov1\dataset")
        self.parser.add_argument("--weight_path", type=str,
                                 default=r"D:\pycharm\yolov1\pretrain\epoch18.pkl",
                                 help="load path for model weight")

        self.opts = self.parser.parse_args()
        if torch.cuda.is_available():
            self.opts.use_GPU = True
            self.opts.GPU_id = torch.cuda.current_device()
            print("use GPU %d to train." % (self.opts.GPU_id))
        else:
            print("use CPU to train.")

    def get_opts(self):
        return self.opts


