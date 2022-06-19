import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import Encoder
from decoder import Decoder


class Discriminator(nn.Module):
    def __init__(self, img_dim=64):
        super(Discriminator, self).__init__()

        self.encoder = Encoder(img_dim=img_dim)
        self.decoder = Decoder(img_dim=img_dim)
    
    def freeze_layers(self):
        self.encoder.freeze_layers()
        self.decoder.freeze_layers()

    def forward(self, in_data, carry=0):
        z = self.encoder(in_data, carry)
        recon = self.decoder(z, carry)
        return recon, z
