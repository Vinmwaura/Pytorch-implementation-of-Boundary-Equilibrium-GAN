import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import Encoder
from decoder import Decoder


class cluster_VAE(nn.Module):
    def __init__(self, img_dim=64, device="cpu"):
        super(cluster_VAE, self).__init__()

        self.encoder = Encoder(img_dim=img_dim, device=device)
        self.decoder = Decoder(img_dim=img_dim, device=device)

    def forward(self, in_data):
        z, mu, sigma = self.encoder(in_data)
        recon = self.decoder(z)
        return recon, z, mu, sigma
