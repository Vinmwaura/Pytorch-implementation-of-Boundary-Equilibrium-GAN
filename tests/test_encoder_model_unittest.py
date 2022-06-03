import unittest

import torch
from encoder import Encoder


class TestEncoderModel(unittest.TestCase):
    def setUp(self):
        self.batch = 64
        self.img_dim = 64
        self.z_dim = 512
        self.test_imgs = torch.ones((self.batch, 3, self.img_dim, self.img_dim))

    def test_encoder(self):
        encoder_net = Encoder(img_dim=self.img_dim)
        self.assertIsNotNone(encoder_net)

        z = encoder_net(self.test_imgs)
        self.assertIsInstance(z.shape, tuple)

        self.assertEqual(z.shape, (self.batch, self.z_dim))
