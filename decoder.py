import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_layers import *


class Decoder(nn.Module):
    def __init__(self, img_dim=64):
        super(Decoder, self).__init__()
        
        min_dim = 4
        max_dim = 1024

        if img_dim <= min_dim or img_dim >= max_dim or img_dim % 4 != 0:
            raise Exception(f"Image dimension must be between {min_dim:,} and {max_dim:,} and be multiple of 4")

        self.decoder_layers = nn.ModuleList()
        
        self.decoder_layers.append(
            nn.Sequential(
                nn.Linear(100, 128),
                nn.ELU(inplace=True),
                nn.Linear(128, 256),
                nn.ELU(inplace=True),
                nn.Linear(256, 512),
                nn.ELU(inplace=True),
                nn.Linear(512, 4*4*64),

                Reshape(-1, 64, 4, 4),
            )
        )

        in_channel = 64
        hidden_channel = 64
        out_channel = 64

        current_dim = 4
        while current_dim < img_dim:
            self.decoder_layers.append(
                nn.Sequential(
                    ConvBlock(
                        in_channel=in_channel,
                        hidden_channel=hidden_channel,
                        out_channel=out_channel),
                    nn.UpsamplingNearest2d(scale_factor=2),
                )
            )
            current_dim *= 2

        self.decoder_layers.append(
            nn.Sequential(
                ConvBlock(
                    in_channel=in_channel,
                    hidden_channel=hidden_channel,
                    out_channel=out_channel),
                ConvBlock_toRGB(
                    in_channel=in_channel,
                    hidden_channel=hidden_channel,
                )

            )
        )

    def forward(self, z):
        x = z
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x)
        return x

