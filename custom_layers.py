import torch
import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class ConvBlock(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        super(ConvBlock, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=hidden_channel,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.ELU(inplace=True),

            nn.Conv2d(
                in_channels=hidden_channel,
                out_channels=out_channel,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, in_data):
        out = self.conv_layer(in_data)
        return out


class ConvBlock_toRGB(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(ConvBlock_toRGB, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=hidden_channel,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(hidden_channel),

            nn.Conv2d(
                in_channels=hidden_channel,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.Tanh()
        )

    def forward(self, in_data):
        out = self.conv_layer(in_data)
        return out


