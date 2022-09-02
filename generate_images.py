import os

import torch
import torchvision

from generator import Generator

from utils import load_checkpoint

def plot_sampled_images(sampled_imgs, file_name, dest_path=None):
    # Convert from BGR to RGB,
    permute = [2, 1, 0]
    sampled_imgs = sampled_imgs[:, permute]

    grid_img = torchvision.utils.make_grid(
        sampled_imgs,
        nrow=5)
    
    if dest_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dir_path = os.path.join(current_dir, "out")
    else:
        dir_path = os.path.join(dest_path, "out")
    
    os.makedirs(dir_path, exist_ok=True)
    try:
        torchvision.utils.save_image(
            grid_img,
            os.path.join(dir_path, str(file_name)))
    except Exception as e:
        print(f"An error occured while plotting reconstructed image: {e}")

def generate_images():
    num_images = 25
    # Add Generator Checkpoint here.
    model_checkpoint = ""
    file_name = "out.jpg"
    img_dim = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rand_noise = None

    gen_net = Generator(img_dim=img_dim).to(device)

    # Load Generator Checkpoints.
    if model_checkpoint is not None:
        status, gen_checkpoint = load_checkpoint(model_checkpoint)
        assert status

        gen_net.load_state_dict(gen_checkpoint["model"])

    if rand_noise is None:
        rand_noise = torch.randn(num_images, 128).to(device)
    
    fake_imgs_plot = gen_net(rand_noise)
    plot_sampled_images(fake_imgs_plot, file_name, dest_path=None)

if __name__ == "__main__":
    generate_images()
