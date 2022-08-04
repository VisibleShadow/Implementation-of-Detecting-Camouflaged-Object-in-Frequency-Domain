import torch
import os
import cv2
import datetime
import sys
import time
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import numpy as np
from apex import amp
import torch
import torch.nn as nn
import torch_dct as dct
import torch.nn.functional as F
import torch_dct as DCT
# import matplotlib.pyplot as plt


class Trainer(object):

    def __init__(self, model, config, model_dir):

        self.model = model
        self.cuda = torch.cuda.is_available()
        self.model_dir = model_dir
        self.epoch = 0
        self.config = config

    def eval(self, val_loader, val_dataset):
        self.model.eval()

        bar_steps = len(val_loader)
        process_bar = ShowProcess(bar_steps)
        total = 0

        save_supervision_path = os.path.join("results", "COD10K")
        if not os.path.exists(save_supervision_path):
            os.makedirs(save_supervision_path)

        for i, data in enumerate(val_loader, 0):

            inputs, ycbcr_image, gts = data
            inputs, ycbcr_image, gts = inputs.cuda().float(), ycbcr_image.cuda().float(), gts.cuda().float()

            num_batchsize = ycbcr_image.shape[0]
            size = ycbcr_image.shape[2]

            ycbcr_image = ycbcr_image.reshape(num_batchsize, 3, size // 8, 8, size // 8, 8).permute(0, 2, 4, 1, 3, 5)
            ycbcr_image = DCT.dct_2d(ycbcr_image, norm='ortho')
            ycbcr_image = ycbcr_image.reshape(num_batchsize, size // 8, size // 8, -1).permute(0, 3, 1, 2)

            with torch.set_grad_enabled(False):

                supervision1, o1, o2, o3, f1, f2, f3, f4 = self.model(inputs, ycbcr_image)

                size = inputs.shape[2]
                supervision1 = torch.nn.functional.interpolate(supervision1, size=(size, size))

                supervision1 = torch.sigmoid(supervision1).clamp(0, 1)
                output = supervision1.detach().cpu().numpy()
                image = output[0, :, :, :]
                image = np.transpose(image, (1, 2, 0))
                image = image[:, :, 0]
                image = image * 255.0
                image = np.round(image)
                image = np.uint8(image)
                cv2.imwrite(
                    os.path.join(save_supervision_path, val_dataset.examples[i]["label_name"]), image)

                total += inputs.size(0)

            process_bar.show_process()
        process_bar.close()

    def load_weights(self, file_path, by_name=False, exclude=None):
        """load the weights from the file_path in CNN model.
        """
        print(file_path)
        checkpoint = torch.load(file_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        self.model.load_state_dict(state_dict)
        print("load weights from {} finished.".format(file_path))


class ShowProcess():

    """
    显示处理进度的类
    调用该类相关函数即可实现处理进度的显示
    """
    i = 0  # 当前的处理进度
    max_steps = 0  # 总共需要处理的次数
    max_arrow = 50  # 进度条的长度

    # 初始化函数，需要知道总共的处理次数
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.i = 0
        # 显示函数，根据当前的处理进度i显示进度
        # 效果为[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%

    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps)  # 计算显示多少个'>'
        num_line = self.max_arrow - num_arrow  # 计算显示多少个'-'
        percent = self.i * 100.0 / self.max_steps  # 计算完成进度，格式为xx.xx%
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']' \
                      + '%.2f' % percent + '%' + '\r'  # 带输出的字符串，'\r'表示不换行回到最左边
        sys.stdout.write(process_bar)  # 这两句打印字符到终端
        sys.stdout.flush()

    def close(self, words='done'):
        print('')
        # print(words)
        self.i = 0
