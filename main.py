"""
Self-contained Minimal implementation of VQ-GAN model.
Paper: https://arxiv.org/abs/2012.09841
Reference: https://compvis.github.io/taming-transformers
Copyright: Do whatever you want. Don't ask me.

"""

import os
import tqdm
import wandb
import time
import torch
import torchvision
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from dataset import SimpleDataset
from model import VQGAN
from loss import VQLPIPSWithDiscriminator

def main():
    # Arguments
    epochs      = 100
    batch_size  = 12
    num_workers = 8
    lr          = 4.5e-6
    val_step    = 500 # validate every 500 iterations
    log_step    = 10  # print logs every 100 steps
    save_step   = 5000 # save model and optimiser
    out         = "./output"
    exp_name    = "simple_VQGAN_v2"
    device      = torch.device("cuda:0") # Avoid complications of multi-GPU for simplicity
    # device      = torch.device("cpu")

    wandb.init(
        entity="project_name",
        project="taming-transformers",
        name=exp_name,
    )
    # wandb = None
    os.makedirs(os.path.join(out, exp_name), exist_ok=True)

    # Create dataloader
    train_dataset = SimpleDataset(
        image_list_file="/path/to/open-images-dataset/train.txt",
        size=256
    )
    valid_dataset = SimpleDataset(
        image_list_file="/path/to/open-images-dataset/test.txt",
        size=256
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, shuffle=True)

    model = VQGAN(resolution=256) # assume image size: (256 x 256)
    loss  = VQLPIPSWithDiscriminator(disc_in_channel=3, disc_start=10000, disc_weight=0.8, codebook_weight=1.0)
    optimiser_autoencoder = torch.optim.Adam(model.parameters(), lr=lr * batch_size, betas=(0.5, 0.9))
    optimiser_discriminator = torch.optim.Adam(loss.discriminator.parameters(), lr=lr * batch_size, betas=(0.5, 0.9))

    # Load model from checkpoint
    if False:
        checkpoint = torch.load("output/simple_VQGAN_v2/ckpt-epoch-001-iter-25000.ckpt")
        model.load_state_dict(checkpoint["model"])
        optimiser_autoencoder.load_state_dict(checkpoint["optimiser_autoencoder"])
        optimiser_discriminator.load_state_dict(checkpoint["optimiser_discriminator"])

    # To device
    model = model.to(device)
    loss = loss.to(device)

    start_epoch = 1
    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        for i_loader, train_data in enumerate(train_loader):
            train_data = train_data.to(device)
            global_step = epoch * len(train_loader) + i_loader
        
            # validation
            if i_loader % val_step == 0:
                model.eval()
                try:
                    valid_data = next(valid_iter)
                except (StopIteration, UnboundLocalError):
                    valid_iter = iter(valid_loader)
                    valid_data = next(valid_iter)
                valid_data = valid_data.to(device, non_blocking=True)
                data_rec = model(valid_data)

                # save image
                grid = torchvision.utils.make_grid(data_rec, nrow=batch_size // 2)
                save_image(grid, os.path.join(out, exp_name, f"image-{epoch:03d}-iter-{i_loader}.jpg"))
                if wandb:
                    wandb.log({"validation": wandb.Image(grid)}, step=global_step)
                
                model.train()
                start_time = time.time() # Ignore validation time

            # training
            data_rec, loss_quantise = model(train_data, train=True)

            # AutoEncoder loss
            loss_ae, loss_info = loss(
                loss_quantise, train_data, data_rec, "generator", global_step,
                last_layer=model.decoder.conv_out.weight, split="train"
            ) 
            if wandb:
                wandb.log({"train/aeloss_step": loss_ae}, step=global_step)
                wandb.log(loss_info, step=global_step)
            
            optimiser_autoencoder.zero_grad()
            loss_ae.backward() # Retain the computation graph for generator
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser_autoencoder.step()

            # Discriminator loss
            loss_disc, loss_info = loss(
                loss_quantise, train_data, data_rec, "discriminator", global_step,
                last_layer=model.decoder.conv_out.weight, split="train"
            )
            if wandb:
                wandb.log({"train/discloss_step": loss_disc}, step=global_step)
                wandb.log(loss_info, step=global_step)

            optimiser_discriminator.zero_grad()
            loss_disc.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser_discriminator.step()

            # Print Logs
            if i_loader % log_step == 0:
                print(f"Epoch: {epoch:03d} | iter: {i_loader:05d} | time: {(time.time() - start_time):.4f}s | loss_ae: {loss_ae.item():.4f} | loss_disc: {loss_disc:.4f}")
                start_time = time.time()

            if i_loader % save_step == 0:
                torch.save({
                    "model": model.state_dict(),
                    "optimiser_autoencoder": optimiser_autoencoder.state_dict(),
                    "optimiser_discriminator": optimiser_discriminator.state_dict(),
                },  os.path.join(out, exp_name, f"ckpt-epoch-{epoch:03d}-iter-{i_loader:05d}.ckpt"))


if __name__ == "__main__":
    main()
