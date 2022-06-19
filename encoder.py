import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_layers import *


class Encoder(nn.Module):
    def __init__(self, img_dim=64):
        super(Encoder, self).__init__()

        min_dim = 8
        max_dim = 1024

        self.in_channel = 64
        self.hidden_channel = 64
        self.out_channel = 64

        if img_dim <= min_dim or img_dim >= max_dim or img_dim % 4 != 0:
            raise Exception(f"Image dimension must be between {min_dim:,} and {max_dim:,} and be multiple of 4")

        self.encoder_layers_init = nn.Sequential(
            ConvBlock(
                in_channel=3,
                hidden_channel=self.hidden_channel,
                out_channel=self.out_channel),
        )

        self.encoder_layers = nn.ModuleList()
        current_dim = 8
        while current_dim < img_dim:
            self.encoder_layers.append(
                nn.Sequential(
                    ConvBlock(
                        in_channel=self.in_channel,
                        hidden_channel=self.hidden_channel,
                        out_channel=self.out_channel),
                    nn.AvgPool2d(2, 2),
                )
            )
            current_dim *= 2

        self.encoding_layer = nn.Sequential(
            ConvBlock(
                in_channel=self.in_channel,
                hidden_channel=self.hidden_channel,
                out_channel=128),
            Reshape(-1, 8*8*128),
            nn.Linear(8*8*128, 512),
        )
    
    def freeze_layers(self):
        for init_layer in self.encoder_layers_init.parameters():
            init_layer.requires_grad = False
    
        for param in self.encoder_layers.parameters():
            param.requires_grad = False
        
        for param in self.encoding_layer.parameters():
            param.requires_grad = False

    def forward(self, z, carry=0):
        x = self.encoder_layers_init(z)
        prev_in = F.max_pool2d(x, kernel_size=2, stride=2)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            
            # Vanishing Residuals.
            x = carry * prev_in + (1 - carry) * x
            x = F.elu(x)

            prev_in = F.max_pool2d(x, kernel_size=2, stride=2)
        z = self.encoding_layer(x)
        return z