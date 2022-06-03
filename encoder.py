import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_layers import *


class Encoder(nn.Module):
    def __init__(self, img_dim=64):
        super(Encoder, self).__init__()

        min_dim = 4
        max_dim = 1024

        in_channel = 128
        hidden_channel = 128
        out_channel = 128

        if img_dim <= min_dim or img_dim >= max_dim or img_dim % 4 != 0:
            raise Exception(f"Image dimension must be between {min_dim:,} and {max_dim:,} and be multiple of 4")

        self.encoder_layers_init = nn.Sequential(
            ConvBlock(
                in_channel=3,
                hidden_channel=hidden_channel,
                out_channel=out_channel),
        )

        self.encoder_layers = nn.ModuleList()
        current_dim = 4
        while current_dim < img_dim:
            self.encoder_layers.append(
                nn.Sequential(
                    ConvBlock(
                        in_channel=in_channel,
                        hidden_channel=hidden_channel,
                        out_channel=out_channel),
                    nn.AvgPool2d(2, 2),
                )
            )
            current_dim *= 2

        self.encoder_layers.append(
            nn.Sequential(
                ConvBlock(
                    in_channel=in_channel,
                    hidden_channel=hidden_channel,
                    out_channel=out_channel),
                Reshape(-1, 4*4*128),
            )
        )
        self.fc_linear = nn.Sequential(
            nn.Linear(4*4*128, 512),
        )

    def forward(self, z):
        x = self.encoder_layers_init(z)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        z = self.fc_linear(x)
        return z
