import unittest

import os
import glob
import torch
from imageloader import CustomDataset


class TestImageLoader(unittest.TestCase):
    def setUp(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(current_dir, "assets/img_64.png")
        self.img_list = glob.glob(img_path)

    def test_image_loading(self):
        dataset = CustomDataset(
            img_list=self.img_list,
            prob=0.2,
            transform=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=True)
        img = next(iter(dataloader))
        self.assertIsNotNone(img)
        self.assertIsInstance(img.shape, tuple)
        self.assertEqual(img.shape, (1, 3, 64, 64))
