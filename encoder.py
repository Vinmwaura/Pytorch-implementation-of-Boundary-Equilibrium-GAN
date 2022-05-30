import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_layers import *


class Encoder(nn.Module):
    def __init__(self, img_dim=64, device="cpu"):
        super(Encoder, self).__init__()

        min_dim = 4
        max_dim = 1024

        in_channel = 64
        hidden_channel = 64
        out_channel = 64

        self.device = device

        if img_dim <= min_dim or img_dim >= max_dim or img_dim % 4 != 0:
            raise Exception(f"Image dimension must be between {min_dim:,} and {max_dim:,} and be multiple of 4")

        self.encoder_layers = nn.ModuleList()
        self.encoder_layers.append(
            nn.Sequential(
                ConvBlock(
                    in_channel=3,
                    hidden_channel=hidden_channel,
                    out_channel=out_channel),
            )
        )

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
                Reshape(-1, 4*4*64),
            )
        )

        self.mu_linear = nn.Sequential(
            nn.Linear(4*4*64, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 100)
        )

        self.sigma_linear = nn.Sequential(
            nn.Linear(4*4*64, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 100)
        )

        self.main_N = torch.distributions.Normal(0, 1)
        if self.device == "cuda":
            self.main_N.loc = self.main_N.loc.cuda()
            self.main_N.scale = self.main_N.scale.cuda()


    def forward(self, z):
        x = z
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        mu = self.mu_linear(x)
        sigma = self.sigma_linear(x)
        normal_sample = self.main_N.rsample((len(mu), 100)).to(self.device)
        z = mu + sigma * normal_sample
        return z, mu, sigma

