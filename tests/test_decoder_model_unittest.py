import unittest

import torch
from decoder import Decoder


class TestDecoderModel(unittest.TestCase):
    def setUp(self):
        self.z = torch.rand((1, 512))
        self.min_dim = 4
        self.max_dim = 1024

    def test_decoder_valid_img_dim(self):
        valid_img_dim = 64
        decoder_net = Decoder(img_dim=valid_img_dim)
        decoder_out = decoder_net(self.z)
        self.assertIsNotNone(decoder_out)
        self.assertIsInstance(decoder_out.shape, tuple)
        self.assertEqual(decoder_out.shape, (1, 3, valid_img_dim, valid_img_dim))
        
    def test_decoder_invalid_img_dim(self):
        invalid_img_dim = 0
        with self.assertRaises(Exception) as context:
            Decoder(img_dim=invalid_img_dim)

        self.assertTrue(f"Image dimension must be between {self.min_dim:,} and {self.max_dim:,} and be multiple of 4" in str(context.exception))

        invalid_img_dim = 2048
        with self.assertRaises(Exception) as context:
            Decoder(img_dim=invalid_img_dim)

        self.assertTrue(f"Image dimension must be between {self.min_dim:,} and {self.max_dim:,} and be multiple of 4" in str(context.exception))
