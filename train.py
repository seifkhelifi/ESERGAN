import torch
import config
from torch import nn
from torch import optim
from utils import gradient_penalty, load_checkpoint, save_checkpoint, plot_examples
from loss import VGGLoss
from torch.utils.data import DataLoader
from model import Generator, Discriminator, initialize_weights
from tqdm import tqdm
from dataset import MyImageFolder
from torch.utils.tensorboard import SummaryWriter

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

