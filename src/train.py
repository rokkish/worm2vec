"""
    Train VAE, save params.
"""
from __future__ import print_function

import torch
import torch.optim as optim
from torchvision import transforms

import argparse

# 自作
from models.vae import VAE
from features.worm_dataset import WormDataset
from features.worm_transform import ToBinary, FillHole, Labelling, Padding, ToNDarray
from visualization.save_images_gray_grid import save_images_grid
import config

# 可視化
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_datasets():
    """ Set transform """
    worm_transforms = transforms.Compose([
        ToBinary(),
        FillHole(),
        Labelling(),
        Padding(),
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor()])

    """ Set dataset """
    train_set = WormDataset(root="F:\Tanimoto_eLife_Fig3B", train=True,
        transform=worm_transforms)
    test_set = WormDataset(root="F:\Tanimoto_eLife_Fig3B", train=False,
        transform=worm_transforms)

    """ Dataloader """
    # Training dataset
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    # Test dataset
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=config.BATCH_SIZE, shuffle=True)

    return train_loader, test_loader

def train(epoch, vae, train_loader, optimizer):

    def loss_function(recon_x, x, mu, logvar):
        BCE = torch.mean((recon_x - x)**2)
        KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
        return BCE, KLD * 0.1

    vae.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, _ = data.to(device), target.to(device)
        optimizer.zero_grad()
        rec, mu, logvar = vae(data)
        loss_re, loss_kl = loss_function(rec, data, mu, logvar)
        (loss_re + loss_kl).backward()
        optimizer.step()

        if batch_idx % (len(train_loader) // 10 + 1) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_re: {:.6f} \tLoss_kl: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_re.item(), loss_kl.item()))

        if batch_idx == 0:
            topil = transforms.ToPILImage()
            tond = ToNDarray()
            img = tond(topil(_.cpu()[0]))
            plt.imsave("input_img.png", img, cmap='gray')

    return loss_re, loss_kl

def evaluate(epoch, vae, test_loader):
    sample_v = torch.randn(256, config.z_size).view(-1, config.z_size, 1, 1)

    with torch.no_grad():
        vae.eval()

        for batch_idx, (x, target) in enumerate(test_loader):
            # Reconstruction from testdata
            x = x.to(device)
            x_rec, _, _ = vae.forward(x)

            x = (x - x.min()) / (x.max() - x.min())
            x_rec = (x_rec - x_rec.min()) / (x_rec.max() - x_rec.min())

            resultsample = torch.cat([x, x_rec])
            resultsample = resultsample.cpu()

            save_images_grid(resultsample, nrow=16, scale_each=True,
                        filename=r'c:\Users\u853337i\Desktop\worm2vec\worm2vec/results/training/{:0=3}_sample_encode.png'.format(epoch))

            # Reconstruction from random (mu, var)
            sample_v = sample_v.to(device)
            x_rec = vae.decode(sample_v)

            resultsample = x_rec * 0.5 + 0.5
            resultsample = resultsample.cpu()

            save_images_grid(resultsample, nrow=16, scale_each=True,
                        filename=r'c:\Users\u853337i\Desktop\worm2vec\worm2vec/results/training/{:0=3}_sample_decode.png'.format(epoch))
            break

def main(args):
    """ load datasets """
    train_loader, test_loader = load_datasets()

    # start tensorboard
    writer = SummaryWriter(log_dir="../log/" + args.log, comment=args.comment)

    # Def model
    vae = VAE(zsize=config.z_size, layer_count=config.layer_count, channels=1)
    vae.to(device)

    optimizer = optim.SGD(vae.parameters(), lr=0.01)

    # Train model
    for epoch in range(1, args.epoch + 1):
        loss_re, loss_kl = train(epoch, vae, train_loader, optimizer)
        #evaluate(epoch, vae, test_loader)

        writer.add_scalar(tag="train_loss/re", scalar_value=loss_re.item(), global_step=epoch)
        writer.add_scalar(tag="train_loss/kl", scalar_value=loss_kl.item(), global_step=epoch)

    # end tensorboard
    writer.close()

    # Save model
    torch.save(vae.state_dict(), "../models/VAEmodel.pkl")

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-e", "--epoch", type=int, default=15)
    parse.add_argument("-m", "--comment", type=str, default="test")
    parse.add_argument("--log", type=str, default="default")
    args = parse.parse_args()

    main(args)
