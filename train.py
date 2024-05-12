import torch
import config
from torch import nn
from torch import optim
from utils import gradient_penalty
from loss import VGGLoss
from torch.utils.data import DataLoader
from model import Generator, Discriminator, initialize_weights
from tqdm import tqdm
from dataset import SRDataset
from config import *
from model import *
from loss import *


torch.backends.cudnn.benchmark = True

def train(
    loader,
    disc,
    gen,
    opt_gen,
    opt_disc,
    l1,
    vgg_loss,
    g_scaler,
    d_scaler,
    tb_step,

    num_epochs=10,
    device='cpu',
    checkpoint_path=None  # Path to the checkpoint file
):
    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        gen.load_state_dict(checkpoint['gen_state_dict'])
        opt_gen.load_state_dict(checkpoint['opt_gen_state_dict'])
        disc.load_state_dict(checkpoint['disc_state_dict'])
        opt_disc.load_state_dict(checkpoint['opt_disc_state_dict'])
        tb_step = checkpoint['tb_step']
        iteration_start = checkpoint['iteration'] + 1
        print("checkpoint loaded sucessfully")
    else:
        iteration_start = 0

    for iteration in range(iteration_start, num_epochs):
        loop = tqdm(loader, leave=True)

        for idx, (low_res, high_res) in enumerate(loop):
            high_res = high_res.to(device)
            low_res = low_res.to(device)

            with torch.cuda.amp.autocast():
                fake = gen(low_res)
                critic_real = disc(high_res)
                critic_fake = disc(fake.detach())
                gp = gradient_penalty(disc, high_res, fake, device=device)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + lambda_gp * gp
                )

            opt_disc.zero_grad()
            d_scaler.scale(loss_critic).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()

            # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            with torch.cuda.amp.autocast():
                l1_loss = 1e-2 * l1(fake, high_res)
                adversarial_loss = 5e-3 * -torch.mean(disc(fake))
                loss_for_vgg = vgg_loss(fake, high_res)
                gen_loss = l1_loss + loss_for_vgg + adversarial_loss

            opt_gen.zero_grad()
            g_scaler.scale(gen_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()
            tb_step += 1

            loop.set_postfix(
                epoch=str(iteration)+"/"+str(num_epochs),
                gp=gp.item(),
                critic=loss_critic.item(),
                l1=l1_loss.item(),
                vgg=loss_for_vgg.item(),
                adversarial=adversarial_loss.item(),
            )

        # Save checkpoint every 50 epochs
        if iteration == 200 or iteration == 220:
            checkpoint_name = fr'checkpoint_epoch_{iteration}.pth'
            checkpoint_path = fr'C:\Users\dell\Desktop\ESRGAN\models\{checkpoint_name}'
            torch.save({
                'gen_state_dict': gen.state_dict(),
                'opt_gen_state_dict': opt_gen.state_dict(),
                'disc_state_dict': disc.state_dict(),
                'opt_disc_state_dict': opt_disc.state_dict(),
                'tb_step': tb_step,
                'iteration': iteration
            }, checkpoint_path)
            

if __name__ == "__main__":
    train_dataset = SRDataset(
        root_dir=r'C:\Users\dell\Desktop\ESRGAN\data\DIV2K_train_HR',
        lowres_transform=lowres_transform,
        highres_transform=highres_transform,
        both_transform=both_transforms
    )

    print(len(train_dataset))

    val_dataset = SRDataset(
        root_dir=r'C:\Users\dell\Desktop\ESRGAN\data\DIV2K_valid_HR',
        lowres_transform=lowres_transform,
        highres_transform=highres_transform,
        both_transform=both_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers
    )

    generator = Generator(in_channels=num_channels, num_blocks=1).to(device)
    discriminator = Discriminator(in_channels=num_channels).to(device)
    init_weights(generator)
    model = {
        'discriminator': discriminator,
        'generator': generator
    }

    optimizer_generator = torch.optim.Adam(model['generator'].parameters(), lr=lr, betas=(0.0, 0.9))
    optimizer_discriminator = torch.optim.Adam(model['discriminator'].parameters(), lr=lr, betas=(0.0, 0.9))
    optimizer = {
        'discriminator': optimizer_discriminator,
        'generator': optimizer_generator
    }

    criterion_generator = ESRGANLoss(l1_weight=1e-2, device=device)
    criterion_discriminator = nn.BCELoss().to(device)
    criterion = {
        'discriminator': criterion_discriminator,
        'generator': criterion_generator
    }

    generator_scaler = torch.cuda.amp.GradScaler()
    discriminator_scaler = torch.cuda.amp.GradScaler()

    tb_step = train(
        loader=train_loader,
        disc=model["discriminator"],
        gen=model['generator'],
        opt_gen=optimizer['generator'],
        opt_disc=optimizer['discriminator'],
        l1=nn.L1Loss(),
        vgg_loss=VGGLoss(device),
        g_scaler=generator_scaler,
        d_scaler=discriminator_scaler,
        tb_step=0,
        device=device,
        num_epochs=220,
        checkpoint_path=fr"C:\Users\dell\Desktop\ESRGAN\checkpoint_epoch_185.pth"
    )
