import os
import cv2
import numpy as np

import torch


def plot_reconstructed(
        in_data,
        recon_data,
        file_name,
        dest_path=None):
    img_h = None
    img_v = None

    for index, (in_data, recon_data) in enumerate(zip(in_data, recon_data)):
        in_data = (np.transpose(in_data, (1, 2, 0)) + 1) / 2
        out_data = (np.transpose(recon_data, (1, 2, 0)) + 1) / 2

        combined_img = np.concatenate((
            in_data,
            out_data), axis=1)

        if index % 5 == 0:
            if img_v is None:
                img_v = img_h
            else:
                img_v = np.concatenate((img_h, img_v), axis=1)
            img_h = None

        if img_h is None:
            img_h = combined_img
        else:
            img_h = np.concatenate((combined_img, img_h), axis=0)
        if index > 25:
            break

    if dest_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dir_path = os.path.join(current_dir, "plots")
    else:
        dir_path = os.path.join(dest_path, "plots")
    os.makedirs(dir_path, exist_ok=True)
    try:
        cv2.imwrite(
            os.path.join(dir_path, str(file_name) + ".jpg"),
            img_v * 255,
            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    except Exception as e:
        print(f"An error occured while plotting reconstructed image. {e}")


def save_model(model_net, file_name, dest_path, checkpoint=False, steps=0):
    try:
        if checkpoint:
            f_path = os.path.join(dest_path, "checkpoint")
        else:
            f_path = os.path.join(dest_path, "models")
        
        os.makedirs(f_path, exist_ok=True)

        model_name = f"{file_name}_{str(steps)}.pt"
        torch.save(
            model_net,
            os.path.join(f_path, model_name))
        return True
    except Exception as e:
        print(f"Exception occured while saving cvae model: {e}.")
        return False

def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)

        k = checkpoint["k"]
        epoch = checkpoint["epoch"]
        global_steps = checkpoint["global_steps"]
        cur_lr = checkpoint["cur_lr"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        print("No checkpoint found.")
    return model, optimizer, epoch, global_steps, cur_lr, k
