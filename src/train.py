"""
    Train VAE, save params.
"""
from __future__ import print_function

import os

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
from tensorboardX import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter
#import matplotlib.pyplot as plt

### begin region ###

import logging

# create logger
logger = logging.getLogger('train_VAE')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
sh = logging.StreamHandler()
fh = logging.FileHandler("../log/logger/test.log")
sh.setLevel(logging.INFO)
fh.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s')

# add formatter to handler
sh.setFormatter(formatter)
fh.setFormatter(formatter)

# add handler to logger
logger.addHandler(sh)
logger.addHandler(fh)

### end region ###

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_processed_datasets():
    """ Set dataset """
    train_set = WormDataset(root="../../data/processed", train=True,
        transform=None, processed=True)

    test_set = WormDataset(root="../../data/processed", train=False,
        transform=None, processed=True)

    """ Dataloader """
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config.BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=config.BATCH_SIZE, shuffle=True)

    return train_loader, test_loader

def train(epoch, vae, train_loader, optimizer, writer):

    def loss_function(recon_x, x, mu, logvar):
        BCE = torch.mean((recon_x - x)**2)
        KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
        return BCE, KLD * 0.1

    vae.train()

    #logger.debug("Load data")
    for batch_idx, (data, target) in enumerate(train_loader):
        logger.debug("Train batch: %d/%d " % (batch_idx + 1, len(train_loader)))

        data = data.to(device)

        optimizer.zero_grad()

        #logger.debug("foward")
        rec, mu, logvar = vae(data)

        #logger.debug("get loss")
        loss_re, loss_kl = loss_function(rec, data, mu, logvar)

        #logger.debug("backward")
        (loss_re + loss_kl).backward()

        optimizer.step()

        logger.debug("Train batch: [{}/{} ({:.0f}%)]\tLoss_re: {:.6f} \tLoss_kl: {:.6f}".format(
                batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_re.item(), loss_kl.item()))

        writer.add_scalar(tag="train_loss/re", scalar_value=loss_re.item(), global_step=batch_idx)
        writer.add_scalar(tag="train_loss/kl", scalar_value=loss_kl.item(), global_step=batch_idx)

    return loss_re, loss_kl

def evaluate(epoch, vae, test_loader, writer):
    if epoch % 10 != 0:
        return

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
                        filename=r'../results/training/{:0=3}_sample_encode.png'.format(epoch))

            # Reconstruction from random (mu, var)
            sample_v = sample_v.to(device)
            x_rec = vae.decode(sample_v)

            resultsample = x_rec * 0.5 + 0.5
            resultsample = resultsample.cpu()

            save_images_grid(resultsample, nrow=16, scale_each=True,
                        filename=r'../results/training/{:0=3}_sample_decode.png'.format(epoch))
            break

    #sample_v = torch.randn(128, config.z_size).view(-1, 1, config.z_size, config.z_size)
    #if epoch == 1:
    #    writer.add_graph(vae, sample_v)

def main(args):
    """ load datasets """
    logger.info("Begin train")
    train_loader, test_loader = load_processed_datasets()

    # start tensorboard
    writer = SummaryWriter(log_dir="../log/tensorboard/" + args.logdir, comment=args.comment)

    # Def model
    logger.info("define model")
    vae = VAE(zsize=config.z_size, layer_count=config.layer_count, channels=1)
    print(vae)
    vae.to(device)

    optimizer = optim.SGD(vae.parameters(), lr=0.01)

    # Train model
    for epoch in range(1, args.epoch + 1):
        
        logger.info("Epoch: %d/%d" % (epoch, args.epoch))
        loss_re, loss_kl = train(epoch, vae, train_loader, optimizer, writer)
        logger.info("End epoch train")

        evaluate(epoch, vae, test_loader, writer)
        logger.info("End epoch evaluation")

        #writer.add_scalar(tag="train_loss/re", scalar_value=loss_re.item(), global_step=epoch)
        #writer.add_scalar(tag="train_loss/kl", scalar_value=loss_kl.item(), global_step=epoch)

    # end tensorboard
    writer.close()

    # Save model
    torch.save(vae.state_dict(), "../models/VAEmodel.pkl")

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-e", "--epoch", type=int, default=15)
    parse.add_argument("-m", "--comment", type=str, default="test")
    parse.add_argument("--logdir", type=str, default="default", help="set path of logfile ../log/tensorboard/")
    parse.add_argument("--gpu_id", type=str, default="0",
                help="When you want to use 1 GPU, input 0. Using Multiple GPU, input [0, 1]")
    args = parse.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)  # choose GPU:0
    main(args)
