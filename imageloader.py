import glob
import cv2
import numpy as np
import torch
import random
import torchvision
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, img_list, prob=0.0, transform=False):
        self.img_list = img_list
        self.transform = transform
        self.transformations = torchvision.transforms.RandomApply(
            torch.nn.ModuleList([
                torchvision.transforms.ColorJitter(),
                torchvision.transforms.RandomErasing(
                    p=prob,
                    scale=(0.02, 0.33),
                    ratio=(0.3, 3.3),
                    value=0,
                    inplace=False)
            ]),
            p=prob)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_paths = self.img_list[index]

        img = cv2.imread(img_paths)
        img = (img.astype(float) - 127.5) / 127.5
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1)
        if self.transform:
            img_tensor = self.transformations(img_tensor)

        return img_tensor

