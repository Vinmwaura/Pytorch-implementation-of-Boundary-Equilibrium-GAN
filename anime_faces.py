import os
import sys
import csv
import glob
import random
import logging

import torch
torch.manual_seed(69)

import torchvision
import torch.nn.functional as F

from torchvision.utils import save_image

from imageloader import CustomDataset

from generator import Generator
from discriminator import Discriminator

from utils import *


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    max_epoch = 1000

    img_dim = 64
    learning_rate = 0.0001
    transform_bool = False
    transform_prob = 0
    batch_size = 128
    dataset_path = ""
    out_dir = ""
    disc_checkpoint = None
    gen_checkpoint = None

    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, f"img_{img_dim}.log")
    logging.basicConfig(filename=log_path, encoding='utf-8', level=logging.DEBUG)

    disc_steps = 1
    gen_steps = 1
    
    gen_net = Generator(img_dim=img_dim).to(device)
    disc_net = Discriminator(img_dim=img_dim).to(device)

    img_regex = os.path.join(dataset_path, "*.jpg")
    img_list = glob.glob(img_regex)

    # Initialize gradient scaler.
    scaler = torch.cuda.amp.GradScaler()

    dataset = CustomDataset(
        img_list=img_list,
        prob=transform_prob,
        transform=transform_bool)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True)

    # Optimizers.
    disc_optim = torch.optim.Adam(
        disc_net.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999))

    gen_optim = torch.optim.Adam(
        gen_net.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999))

    z_noise_plot = torch.randn(25, 100)

    # BEGAN Params.
    k = 0
    gamma = 0.7
    total_convergence_score = 0
    lambda_val = 1e-3

    global_steps = 0
    checkpoint_steps = 1000

    starting_epoch = 0

    # Load Generator Checkpoints.
    if gen_checkpoint is not None and os.path.exists(gen_checkpoint):
        gen_net, gen_optim, starting_epoch, global_steps, learning_rate, k = load_checkpoint(
            gen_net,
            gen_optim,
            checkpoint_path=gen_checkpoint)
    # Load Discriminator Checkpoints.
    if disc_checkpoint is not None and os.path.exists(disc_checkpoint):
        disc_net, disc_optim, _, _, _, _ = load_checkpoint(
            disc_net,
            disc_optim,
            checkpoint_path=disc_checkpoint)

    cur_lr = learning_rate

    logging.info(f"disc BeGAN Parameters:")
    logging.info(f"Img dim: {img_dim:,}")
    logging.info(f"Init Learning Rate: {learning_rate:.5f}")
    logging.info(f"Max Epoch: {max_epoch:,}")
    logging.info(f"disc Steps: {disc_steps}")
    logging.info(f"Generator Steps: {gen_steps}")
    logging.info(f"Transform: {transform_bool}")
    logging.info(f"Augment Probability: {transform_prob:,.2f}")
    logging.info(f"Batch size: {batch_size:,}")
    logging.info(f"Dataset Path: {dataset_path}")
    logging.info(f"Output Path: {out_dir}")
    logging.info(f"Gamma: {gamma:.3f}")
    logging.info(f"BEGAN Lambda: {lambda_val:.9f}")
    logging.info(f"Checkpoint Steps: {checkpoint_steps}")
    logging.info("*" * 100)

    for epoch in range(starting_epoch, max_epoch):
        # Generator.
        total_gen_loss = 0
        
        # C_VAE.
        total_disc_loss = 0
        total_convergence_score = 0

        # Counters.
        gen_count = 0
        disc_count = 0

        for index, tr_data in enumerate(dataloader):
            #################################################
            #             Cluster VAE Training.             #
            #################################################
            if index % disc_steps == 0:
                disc_count += 1

                # Real Images Transformations and Real Losses.
                tr_data = tr_data.to(device)

                with torch.no_grad():
                    # Fake Images Generation.
                    z_noise = torch.randn((len(tr_data), 100)).to(device)
                    fake_imgs = gen_net(z_noise)

                del z_noise
                torch.cuda.empty_cache()

                disc_optim.zero_grad()

                # Enable autocasting for mixed precision.
                with torch.cuda.amp.autocast():
                    # Real Images Reconstruction.
                    x_hat_real, z_real = disc_net(tr_data)
                    recon_loss_real = F.mse_loss(
                        x_hat_real,
                        tr_data,
                        reduction="mean")

                    # Generated Images Reconstruction.
                    x_hat_fake, z_fake = disc_net(fake_imgs)
                    recon_loss_fake = F.mse_loss(
                        x_hat_fake,
                        fake_imgs,
                        reduction="mean")
                    began_loss_D = recon_loss_real - (k * recon_loss_fake)

                # Scale the loss and do backprop.
                scaler.scale(began_loss_D).backward()

                # Update the scaled parameters.
                scaler.step(disc_optim)

                # Update the scaler for next iteration
                scaler.update()

                total_disc_loss += began_loss_D.item()

                # Update K value.
                balance = (gamma * recon_loss_real.detach().item()) - recon_loss_fake.detach().item()
                k = k + (lambda_val * balance)
                k = max(min(1, k), 0)
                total_convergence_score += (recon_loss_real.detach().item() + abs(balance))

            #################################################
            #               Generator Training.             #
            #################################################
            if index % gen_steps == 0:
                gen_count += 1

                gen_optim.zero_grad()

                # Enable autocasting for mixed precision.
                with torch.cuda.amp.autocast():
                    # Generated Images Reconstruction.
                    z_noise = torch.randn((len(tr_data), 100)).to(device)
                    fake_imgs = gen_net(z_noise)

                    x_hat_fake, z_fake = disc_net(fake_imgs)
                    recon_loss_fake = F.mse_loss(
                        x_hat_fake,
                        fake_imgs,
                        reduction="mean")

                    fake_loss = recon_loss_fake

                # Scale the loss and do backprop.
                scaler.scale(fake_loss).backward()

                # Update the scaled parameters.
                scaler.step(gen_optim)

                # Update the scaler for next iteration
                scaler.update()

                total_gen_loss += fake_loss.item()

            # Checkpoint and Plot Images.
            if global_steps % checkpoint_steps == 0 and global_steps >= 0:
                disc_state = {
                    "epoch": epoch,
                    "model": disc_net.state_dict(),
                    "optimizer": disc_optim.state_dict(),
                    "cur_lr": cur_lr,
                    "k": k,
                    "global_steps": global_steps
                }

                # Save Disc Net.
                save_model(
                    model_net=disc_state,
                    file_name="disc",
                    dest_path=out_dir,
                    checkpoint=True,
                    steps=global_steps)

                gen_state = {
                    "epoch": epoch,
                    "model": gen_net.state_dict(),
                    "optimizer": gen_optim.state_dict(),
                    "cur_lr": cur_lr,
                    "k": k,
                    "global_steps": global_steps
                }

                # Save Gen Net.
                save_model(
                    model_net=gen_state,
                    file_name="gen",
                    dest_path=out_dir,
                    checkpoint=True,
                    steps=global_steps)

                # Real Recon.
                if batch_size > 25:
                    with torch.no_grad():
                        x_hat_real_plot, _ = disc_net(tr_data)
                    plot_reconstructed(
                        in_data=tr_data.detach().cpu().numpy(),
                        recon_data=x_hat_real_plot.detach().cpu().numpy(),
                        file_name=f"began_real_{global_steps}",
                        dest_path=out_dir)

                # Fake Recon.
                with torch.no_grad():
                    z_noise_plot = z_noise_plot.to(device)
                    fake_imgs_plot = gen_net(z_noise_plot)
                    x_hat_real_plot, _ = disc_net(fake_imgs_plot)
                plot_reconstructed(
                    in_data=fake_imgs_plot.detach().cpu().numpy(),
                    recon_data=x_hat_real_plot.detach().cpu().numpy(),
                    file_name=f"began_fake_{global_steps}",
                    dest_path=out_dir)

            temp_avg_gen = total_gen_loss / gen_count
            temp_avg_disc = total_disc_loss / disc_count
            temp_avg_convergence = total_convergence_score / disc_count

            message = f"Cum. Steps: {global_steps + 1} | Steps: {index + 1} / {len(dataloader)} | Disc Loss: {temp_avg_disc:,.5f} | Gen Loss: {temp_avg_gen:,.5f} | Convergence: {temp_avg_convergence:,.5f} | k: {k:.5f} | balance: {balance:,.5f}"
            logging.info(message)
            print(message)
            sys.stdout.write("\033[F") #  Back to previous line 
            sys.stdout.write("\033[K") #  Clear line
            global_steps += 1

        # Save Model every epoch as well
        disc_state = {
            "epoch": epoch,
            "model": disc_net.state_dict(),
            "optimizer": disc_optim.state_dict(),
            "cur_lr": cur_lr,
            "k": k,
            "global_steps": global_steps
        }
        # Save disc Net.
        save_model(
            model_net=disc_state,
            file_name="disc",
            dest_path=out_dir,
            checkpoint=True,
            steps=global_steps)

        gen_state = {
            "epoch": epoch,
            "model": gen_net.state_dict(),
            "optimizer": gen_optim.state_dict(),
            "cur_lr": cur_lr,
            "k": k,
            "global_steps": global_steps
        }
        # Save Gen Net.
        save_model(
            model_net=gen_state,
            file_name="gen",
            dest_path=out_dir,
            checkpoint=True,
            steps=global_steps)

        # Update Learning Rate every 100 epochs.
        lr = learning_rate * 0.95**(global_steps // 3000)
        for p in disc_optim.param_groups + gen_optim.param_groups:
            p['lr'] = lr
            cur_lr = lr

        avg_gen_loss = total_gen_loss / gen_count
        avg_disc_loss = total_disc_loss / disc_count
        avg_convergence = total_convergence_score / disc_count
        message = f"Epoch: {epoch + 1} / {max_epoch} | Disc: {avg_disc_loss:,.2f} | Gen: {avg_gen_loss:,.2f} | Convergence: {avg_convergence:,.3f} | lr: {p['lr']:.8f}"
        logging.info(message)
        print(message)

if __name__ == '__main__':
    main()
