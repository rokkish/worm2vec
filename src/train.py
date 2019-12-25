"""
    Train VAE, save params.
"""
from __future__ import print_function

import torch
import torch.optim as optim
from torchvision import transforms

from models.vae import VAE
from features.worm_dataset import WormDataset
from features.worm_transform import ToBinary, FillHole, ToNDarray
import config

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_datasets():
    """ Set transform """
    worm_transforms = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        ToBinary(),
#        FillHole(),
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

def train(epoch, vae, train_loader, test_loader, optimizer):

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

        if batch_idx % (config.BATCH_SIZE // 2) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_re: {:.6f} \tLoss_kl: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_re.item(), loss_kl.item()))

        if batch_idx == 0:
            topil = transforms.ToPILImage()
            tond = ToNDarray()
            img = tond(topil(_.cpu()[0]))
            plt.imsave("input_img.png", img, cmap='gray')


def main():
    """ load datasets """
    train_loader, test_loader = load_datasets()

    """ Def model"""
    vae = VAE(zsize=config.z_size, layer_count=config.layer_count, channels=1)
    if device == "cuda":
        print(vae.cuda())
    vae.to(device)

    optimizer = optim.SGD(vae.parameters(), lr=0.01)

    """ Train model """
    for epoch in range(1, 15 + 1):
        train(epoch, vae, train_loader, test_loader, optimizer)
        #test()

    """ Save model """
    torch.save(vae.state_dict(), "../models/VAEmodel.pkl")

if __name__ == "__main__":
    main()