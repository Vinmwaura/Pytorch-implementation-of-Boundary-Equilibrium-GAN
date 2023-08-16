import cv2

import torch

import torchvision
from torch.utils.data import Dataset


from custom_transforms import *

class CustomDataset(Dataset):
    def __init__(self, img_list, prob=0.0, transform=False, device="cpu"):
        self.img_list = img_list
        self.transform = transform
        self.transformations = torchvision.transforms.RandomApply(
            torch.nn.ModuleList([
                torchvision.transforms.ColorJitter(),
                torchvision.transforms.GaussianBlur(
                    kernel_size=5,
                    sigma=(0.1, 2.0)),
                torchvision.transforms.RandomInvert(p=0.5),
                torchvision.transforms.RandomSolarize(threshold=0.75, p=0.5),
                torchvision.transforms.RandomAdjustSharpness(sharpness_factor=20, p=0.5),
                AddGaussianNoise(device, mean=0., std=0.3),
                torchvision.transforms.RandomErasing(
                    p=0.5,
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
            img_transformer = self.transformations(img_tensor)
            return img_transformer
        else:
            return img_tensor
