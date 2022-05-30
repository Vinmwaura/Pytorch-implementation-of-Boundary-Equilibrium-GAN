import unittest

import os
import shutil
import numpy as np

from utils import plot_reconstructed


class TestReconImages(unittest.TestCase):
    def setUp(self):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.origin_plot_imgs = np.ones((25, 3, 64, 64))
        self.recon_plot_imgs = np.zeros((25, 3, 64, 64))
        self.img_path = os.path.join(self.current_dir, "plots/test_img.jpg")

    def tearDown(self):
        shutil.rmtree(
            os.path.join(self.current_dir, "plots"),
            ignore_errors=False,
            onerror=None)

    def test_plot_images(self):
        plot_reconstructed(
            self.origin_plot_imgs,
            self.recon_plot_imgs,
            file_name="test_img",
            dest_path=self.current_dir)
        self.assertTrue(os.path.exists(self.img_path))
