import torch
import torch.nn as nn


class AddGaussianNoise(nn.Module):
    def __init__(self, device="cpu", mean=0., std=1.):
        super(AddGaussianNoise, self).__init__()

        self.std = std
        self.mean = mean
        self.device = device
        
    def forward(self, x):
        return x + torch.randn(x.size()).to(self.device) * self.std + self.mean

