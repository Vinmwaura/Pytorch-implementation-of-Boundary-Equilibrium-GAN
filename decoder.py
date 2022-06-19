import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_layers import *


class Decoder(nn.Module):
    def __init__(self, img_dim=64):
        super(Decoder, self).__init__()
        
        min_dim = 8
        max_dim = 1024

        self.img_dim = img_dim

        if img_dim <= min_dim or img_dim >= max_dim or img_dim % 4 != 0:
            raise Exception(f"Image dimension must be between {min_dim:,} and {max_dim:,} and be multiple of 4")

        self.decoder_layers = nn.ModuleList()
        
        self.decoder_layer_init = nn.Sequential(
            nn.Linear(512, 128*8*8),

            Reshape(-1, 128, 8, 8),
        )

        self.in_channel = 256
        self.hidden_channel = 128
        self.out_channel = 128

        current_dim = 8
        while current_dim < img_dim:
            self.decoder_layers.append(
                nn.Sequential(
                    ConvBlock(
                        in_channel=self.in_channel//2 if current_dim == 8 else self.in_channel,
                        hidden_channel=self.hidden_channel,
                        out_channel=self.out_channel),
                    nn.UpsamplingNearest2d(scale_factor=2),
                )
            )
            current_dim *= 2

        self.decoder_layer_out = nn.Sequential(
            ConvBlock(
                in_channel=self.in_channel,
                hidden_channel=self.hidden_channel,
                out_channel=self.out_channel),
            ConvBlock_toRGB(
                in_channel=self.in_channel//2,
                hidden_channel=self.hidden_channel,
            )
        )
    
    def freeze_layers(self):
        for init_layer in self.decoder_layer_init.parameters():
            init_layer.requires_grad = False
    
        for param in self.decoder_layers.parameters()[:-2]:
            param.requires_grad = False

    def forward(self, z, carry=0):
        init_x = self.decoder_layer_init(z)
        x = init_x
        prev_in = F.interpolate(x, scale_factor=2)
        for index, decoder_layer in enumerate(self.decoder_layers):
            x = decoder_layer(x)
            
            # Vanishing Residuals.
            x = carry * prev_in + (1 - carry) * x
            x = F.elu(x)

            prev_in = F.interpolate(x, scale_factor=2)

            # Skip Connection
            skip_in = F.interpolate(init_x, scale_factor=2**(index + 1))
            x = torch.cat((x, skip_in), dim=1)

        recon = self.decoder_layer_out(x)
        return recon
