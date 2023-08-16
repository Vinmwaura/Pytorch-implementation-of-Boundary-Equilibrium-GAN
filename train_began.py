import os
import glob
import logging

import torch
torch.manual_seed(69)

import torchvision

import torch.nn.functional as F

from imageloader import CustomDataset

from generator import Generator
from discriminator import Discriminator

from utils import *
from custom_transforms import *


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    max_epoch = 1000
    img_dim = 128
    learning_rate = 0.0001 * (0.5**0)
    transform_bool = False
    transform_prob = 0.7
    batch_size = 32
    
    # Regex to image dataset e.g /file/path/to/*.jpg
    dataset_path = ""

    # Output path for generated images.
    out_dir = ""

    disc_checkpoint = None
    gen_checkpoint = None
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, f"{img_dim}.log")
    logging.basicConfig(
        # filename=log_path,
        format="%(asctime)s [%(levelname)s] %(message)s",
        encoding='utf-8',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ],
        level=logging.DEBUG)

    disc_steps = 1
    gen_steps = 1
    
    gen_net = Generator(img_dim=img_dim).to(device)
    disc_net = Discriminator(img_dim=img_dim).to(device)
    img_list = glob.glob(dataset_path)

    # Initialize gradient scaler.
    scaler = torch.cuda.amp.GradScaler()

    dataset = CustomDataset(
        img_list=img_list,
        prob=transform_prob,
        transform=transform_bool,
        device=device)
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

    disc_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=disc_optim,
        mode="min",
        patience=10000,
        threshold=1e-5,
        factor=0.5)
    gen_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=gen_optim,
        mode="min",
        patience=10000,
        threshold=1e-5,
        factor=0.5)

    z_noise_plot = torch.randn(25, 128)

    # Augmentation Functions.
    real_noise_transformer = AddGaussianNoise(device, mean=0., std=1.)

    # BEGAN Params.
    k = 0
    gamma = 1.0  # Diversity Indicator
    total_convergence_score = 0
    lambda_k = 1e-3
    lambda_noise = 2
    
    # Vanishing Residuals.
    carry = 1

    global_steps = 0
    checkpoint_steps = 1000

    starting_epoch = 0

    # Load Generator Checkpoints.
    if gen_checkpoint is not None:
        status, gen_checkpoint = load_checkpoint(gen_checkpoint)
        assert status

        gen_net.load_state_dict(gen_checkpoint["model"])
        gen_optim.load_state_dict(gen_checkpoint["optimizer"])
        gen_lr_scheduler.load_state_dict(gen_checkpoint["scheduler"])
        
        starting_epoch = gen_checkpoint["epoch"]
        global_steps = gen_checkpoint["global_steps"]
        k = gen_checkpoint["k"]

    # Load Discriminator Checkpoints.
    if disc_checkpoint is not None:
        status, disc_checkpoint = load_checkpoint(disc_checkpoint)
        assert status

        disc_net.load_state_dict(disc_checkpoint["model"])
        disc_optim.load_state_dict(disc_checkpoint["optimizer"])
        disc_lr_scheduler.load_state_dict(disc_checkpoint["scheduler"])

    logging.info(f"BeGAN Parameters:")
    logging.info(f"Img dim: {img_dim:,}")
    logging.info(f"Init gen Learning Rate: {gen_optim.param_groups[0]['lr']:.5f}")
    logging.info(f"Init disc Learning Rate: {disc_optim.param_groups[0]['lr']:.5f}")
    logging.info(f"Max Epoch: {max_epoch:,}")
    logging.info(f"disc Steps: {disc_steps}")
    logging.info(f"Generator Steps: {gen_steps}")
    logging.info(f"Transform: {transform_bool}")
    logging.info(f"Augment Probability: {transform_prob:,.2f}")
    logging.info(f"Batch size: {batch_size:,}")
    logging.info(f"Dataset Path: {dataset_path}")
    logging.info(f"Output Path: {out_dir}")
    logging.info(f"Gamma: {gamma:.3f}")
    logging.info(f"BEGAN Lambda: {lambda_k:.9f}")
    logging.info(f"Checkpoint Steps: {checkpoint_steps}")
    logging.info("*" * 100)

    for epoch in range(starting_epoch, max_epoch):
        # Generator.
        total_gen_loss = 0
        
        # Discriminator.
        total_disc_loss = 0
        total_convergence_score = 0

        # Counters.
        gen_count = 0
        disc_count = 0

        for index, tr_data in enumerate(dataloader):
            #################################################
            #             Discriminator Training.             #
            #################################################
            if index % disc_steps == 0:
                disc_count += 1
                
                # Real Images Transformations and Real Losses.
                tr_data = tr_data.to(device)

                with torch.no_grad():
                    # Fake Images Generation.
                    z_noise_disc = torch.randn((len(tr_data), 128)).to(device)
                    fake_imgs = gen_net(z_noise_disc, carry)

                del z_noise_disc
                torch.cuda.empty_cache()
                
                disc_optim.zero_grad()
                
                # Enable autocasting for mixed precision.
                with torch.cuda.amp.autocast():
                    # Real Images Reconstruction: L(x)
                    x_hat_real, z_real = disc_net(tr_data, carry)
                    recon_loss_real_disc = F.mse_loss(
                        x_hat_real,
                        tr_data,
                        reduction="mean")

                    # Denoising Loss.
                    noisy_tr_data = real_noise_transformer(tr_data)
                    noisy_x_hat_real, _ = disc_net(noisy_tr_data, carry)
                    denoising_loss = F.mse_loss(
                        noisy_x_hat_real,
                        tr_data,
                        reduction="mean")

                    # Generated Images Reconstruction: L(G(z_D)).
                    x_hat_fake, z_fake = disc_net(fake_imgs, carry)
                    recon_loss_fake_disc = F.mse_loss(
                        x_hat_fake,
                        fake_imgs,
                        reduction="mean")

                    began_loss_D = (recon_loss_real_disc - (k * recon_loss_fake_disc)) + (lambda_noise * denoising_loss)

                # Scale the loss and do backprop.
                scaler.scale(began_loss_D).backward()

                # Update the scaled parameters.
                scaler.step(disc_optim)

                # Update the scaler for next iteration
                scaler.update()
                
                total_disc_loss += began_loss_D.item()

                real_loss_disc = recon_loss_real_disc.detach().item()

            #################################################
            #               Generator Training.             #
            #################################################
            if index % gen_steps == 0:
                gen_count += 1

                gen_optim.zero_grad()
                
                # Enable autocasting for mixed precision.
                with torch.cuda.amp.autocast():
                    # Fake Images Generation.
                    z_noise_gen = torch.randn((len(tr_data), 128)).to(device)
                    fake_imgs = gen_net(z_noise_gen, carry)

                    x_hat_fake, _ = disc_net(fake_imgs, carry)
                    recon_loss_fake_gen = F.mse_loss(
                        x_hat_fake,
                        fake_imgs,
                        reduction="mean")

                # Scale the loss and do backprop.
                scaler.scale(recon_loss_fake_gen).backward()

                # Update the scaled parameters.
                scaler.step(gen_optim)

                # Update the scaler for next iteration
                scaler.update()

                fake_loss_gen = recon_loss_fake_gen.detach().item()

                total_gen_loss += fake_loss_gen
            
             # Update K value.
            balance = (gamma * real_loss_disc) - fake_loss_gen
            k = k + (lambda_k * balance)
            k = max(min(1, k), 0)

            # Global Convergence Loss.
            convergence = recon_loss_real_disc + torch.abs((gamma * recon_loss_real_disc) - recon_loss_fake_gen)
            total_convergence_score += convergence.detach().item()

            disc_lr_scheduler.step(convergence)
            gen_lr_scheduler.step(convergence)
            
            # Checkpoint and Plot Images.
            if global_steps % checkpoint_steps == 0 and global_steps >= 0:
                disc_state = {
                    "epoch": epoch,
                    "model": disc_net.state_dict(),
                    "optimizer": disc_optim.state_dict(),
                    "scheduler": disc_lr_scheduler.state_dict(),
                    "k": k,
                    "global_steps": global_steps}

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
                    "scheduler": gen_lr_scheduler.state_dict(),
                    "k": k,
                    "global_steps": global_steps}

                # Save Gen Net.
                save_model(
                    model_net=gen_state,
                    file_name="gen",
                    dest_path=out_dir,
                    checkpoint=True,
                    steps=global_steps)
                
                # Real Recon.
                if batch_size > 25:
                    plot_reconstructed(
                        in_data=tr_data.detach().cpu().numpy(),
                        recon_data=x_hat_real.detach().cpu().numpy(),
                        file_name=f"began_real_{global_steps}",
                        dest_path=out_dir)
                
                # Fake Recon.
                with torch.no_grad():
                    z_noise_plot = z_noise_plot.to(device)
                    fake_imgs_plot = gen_net(z_noise_plot, carry)
                    x_hat_fake_plot, _ = disc_net(fake_imgs_plot, carry)
                
                plot_reconstructed(
                    in_data=fake_imgs_plot.detach().cpu().numpy(),
                    recon_data=x_hat_fake_plot.detach().cpu().numpy(),
                    file_name=f"began_fake_{global_steps}",
                    dest_path=out_dir)

            if global_steps % 1000 == 0 and global_steps > 0:
                carry = round(carry - 0.05, 2)
                carry = max(0., carry)

            temp_avg_gen = total_gen_loss / gen_count
            temp_avg_disc = total_disc_loss / disc_count
            temp_avg_convergence = total_convergence_score / disc_count

            message = f"Cum. Steps: {global_steps + 1} | Steps: {index + 1} / {len(dataloader)} | Disc: {temp_avg_disc:,.5f} | Gen: {temp_avg_gen:,.5f} | Convergence: {temp_avg_convergence:,.5f} | k: {k:.5f} | balance: {balance:,.5f} | lr: {gen_optim.param_groups[0]['lr']:.15f} | carry: {carry:.2f}"
            logging.info(message)

            global_steps += 1

        # Save Model every epoch as well
        disc_state = {
            "epoch": epoch,
            "model": disc_net.state_dict(),
            "optimizer": disc_optim.state_dict(),
            "scheduler": disc_lr_scheduler.state_dict(),
            "k": k,
            "global_steps": global_steps}
        
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
            "scheduler": gen_lr_scheduler.state_dict(),
            "k": k,
            "global_steps": global_steps}
        
        # Save Gen Net.
        save_model(
            model_net=gen_state,
            file_name="gen",
            dest_path=out_dir,
            checkpoint=True,
            steps=global_steps)

        avg_gen_loss = total_gen_loss / gen_count
        avg_disc_loss = total_disc_loss / disc_count
        avg_convergence = total_convergence_score / disc_count
        message = f"Epoch: {epoch + 1} / {max_epoch} | Disc: {avg_disc_loss:,.5f} | Gen: {avg_gen_loss:,.5f} | Convergence: {avg_convergence:,.5f} | lr: {gen_optim.param_groups[0]['lr']:.15f}"
        logging.info(message)
        
if __name__ == '__main__':
    main()
