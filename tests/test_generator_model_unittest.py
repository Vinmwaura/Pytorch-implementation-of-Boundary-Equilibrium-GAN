import unittest

import torch
from generator import Generator


class TestGeneratorModel(unittest.TestCase):
    def setUp(self):
        self.z = torch.rand((1, 100))
        self.min_dim = 4
        self.max_dim = 1024

    def test_generate_valid_img_dim(self):
        valid_img_dim = 64
        gen_net = Generator(img_dim=valid_img_dim)
        gen_out = gen_net(self.z)
        self.assertIsNotNone(gen_out)
        self.assertIsInstance(gen_out.shape, tuple)
        self.assertEqual(gen_out.shape, (1, 3, valid_img_dim, valid_img_dim))
        
    def test_generate_invalid_img_dim(self):
        invalid_img_dim = 0
        with self.assertRaises(Exception) as context:
            Generator(img_dim=invalid_img_dim)

        self.assertTrue(f"Image dimension must be between {self.min_dim:,} and {self.max_dim:,} and be multiple of 4" in str(context.exception))

        invalid_img_dim = 2048
        with self.assertRaises(Exception) as context:
            Generator(img_dim=invalid_img_dim)

        self.assertTrue(f"Image dimension must be between {self.min_dim:,} and {self.max_dim:,} and be multiple of 4" in str(context.exception))
