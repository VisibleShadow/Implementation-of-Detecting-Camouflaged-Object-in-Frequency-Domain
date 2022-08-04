import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2


class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        image = (image - self.mean) / self.std
        mask /= 255
        return image, mask


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask


class ToTensor(object):
    def __call__(self, image, ycbcr_image, mask):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        ycbcr_image = torch.from_numpy(ycbcr_image)
        ycbcr_image = ycbcr_image.permute(2, 0, 1)
        mask = torch.from_numpy(mask)
        mask = torch.unsqueeze(mask, 0)
        return image, ycbcr_image, mask


class DatasetVal(Dataset):
    def __init__(self):

        self.mean = np.array([[[124.55, 118.90, 102.94]]])
        self.std = np.array([[[56.77, 55.97, 57.50]]])

        self.normalize = Normalize(mean=self.mean, std=self.std)
        self.resize = Resize(352, 352)
        self.totensor = ToTensor()

        self.img_dir = 'G:/DataSet/COD10K-v3/Test/Image/'
        self.label_dir = 'G:/DataSet/COD10K-v3/Test/GT_Object/'
        # self.img_dir = '/CAMO/Images/Test/'
        # self.label_dir = '/CAMO/GT/'
        # self.img_dir = '/COD10K/Test/Image/'
        # self.label_dir = '/COD10K/Test/GT_Object/'
        # self.img_dir = '/CHAMELEON_TestingDataset/Image/'
        # self.label_dir = '/CHAMELEON_TestingDataset/GT/'
        self.examples = []

        file_names = os.listdir(self.img_dir)

        for file_name in file_names:
            if file_name.find(".jpg") != -1 and file_name.find('NonCAM') == -1:
                img_path = self.img_dir + file_name
                label_img_path = self.label_dir + file_name.replace(".jpg", ".png")

                example = {}
                example["img_path"] = img_path
                example["label_img_path"] = label_img_path
                example["label_name"] = file_name.replace(".jpg", ".png")

                self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, idx):

        example = self.examples[idx]

        img_path = example["img_path"]
        image = cv2.imread(img_path)[:,:,::-1].astype(np.float32)

        label_img_path = example["label_img_path"]
        mask = cv2.imread(label_img_path, 0).astype(np.float32)

        image, mask = self.resize(image, mask)
        ycbcr_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        image, mask = self.normalize(image, mask)
        image, ycbcr_image, mask = self.totensor(image, ycbcr_image, mask)

        return image, ycbcr_image, mask

    def __len__(self):
        return self.num_examples
