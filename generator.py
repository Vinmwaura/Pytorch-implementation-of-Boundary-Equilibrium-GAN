import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_layers import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Generator(nn.Module):
    def __init__(self, img_dim=64):
        super(Generator, self).__init__()
        
        min_dim = 4
        max_dim = 1024

        if img_dim <= min_dim or img_dim >= max_dim or img_dim % 4 != 0:
            raise Exception(f"Image dimension must be between {min_dim:,} and {max_dim:,} and be multiple of 4")

        self.generator_layers = nn.ModuleList()
        
        self.generator_layers_init = nn.Sequential(
            nn.Linear(100, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 1024),
            nn.ELU(inplace=True),
            nn.Linear(1024, 4*4*128),

            Reshape(-1, 128, 4, 4),
        )

        in_channel = 128
        hidden_channel = 128
        out_channel = 128

        current_dim = 4
        while current_dim < img_dim:
            self.generator_layers.append(
                nn.Sequential(
                    ConvBlock(
                        in_channel=in_channel,
                        hidden_channel=hidden_channel,
                        out_channel=out_channel),
                    nn.UpsamplingNearest2d(scale_factor=2),
                )
            )
            current_dim *= 2

        self.generator_layers_out = nn.Sequential(
            ConvBlock(
                in_channel=in_channel,
                hidden_channel=hidden_channel,
                out_channel=out_channel),
            ConvBlock_toRGB(
                in_channel=in_channel,
                hidden_channel=hidden_channel,
            )

        )

    def forward(self, z):
        x = self.generator_layers_init(z)
        for index, generator_layer in enumerate(self.generator_layers):
            prev_in = F.interpolate(x, scale_factor=2)
            x = generator_layer(x)
            x = F.elu(x + prev_in)

        recon = self.generator_layers_out(x)
        return recon
