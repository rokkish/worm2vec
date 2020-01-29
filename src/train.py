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

def load_processed_datasets(train_dir):
    """ Set dataset """
    train_set = WormDataset(root="../../data/"+train_dir, train=True,
        transform=None, processed=True)

    test_set = WormDataset(root="../../data/"+train_dir, train=False,
        transform=None, processed=True)

    """ Dataloader """
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config.BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=config.BATCH_SIZE, shuffle=True)

    return train_loader, test_loader

def train(epoch, vae, train_loader, optimizer, writer, device):

    def loss_function(recon_x, x, mu, logvar):
        BCE = torch.mean((recon_x - x)**2)
        KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
        return BCE, KLD * 0.1

    def rotation_invariance_regularization(target, vae, z):
        RI = 0
        for theta in range(target.shape[1]):
            mu, logvar = vae.encode(target[:, theta])
            mu, logvar = mu.squeeze(), logvar.squeeze()
            z_theta = vae.reparameterize(mu, logvar)
            RI += torch.norm(z - z_theta)
        return RI / target.shape[0] / target.shape[1] * config.lambda_ri

    vae.train()

    #logger.debug("Load data")
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        #logger.debug("forward")
        rec, mu, logvar = vae(data)
        z = vae.reparameterize(mu, logvar)

        #logger.debug("get loss")
        loss_re, loss_kl = loss_function(rec, data, mu, logvar)
        loss_ri = rotation_invariance_regularization(target, vae, z)

        #logger.debug("backward")
        (loss_re + loss_kl + loss_ri).backward()

        optimizer.step()

        if batch_idx % (len(train_loader) // 10) == 0:
            logger.debug("Train batch: [{}/{} ({:.0f}%)]\tLoss_re: {:.5f} \tLoss_kl: {:.5f} \tLoss_ri: {:.5f}".format(
                    batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_re.item(), loss_kl.item(), loss_ri.item()))

        writer.add_scalar(tag="train_loss/re", scalar_value=loss_re.item(), global_step=batch_idx)
        writer.add_scalar(tag="train_loss/kl", scalar_value=loss_kl.item(), global_step=batch_idx)
        writer.add_scalar(tag="train_loss/ri", scalar_value=loss_ri.item(), global_step=batch_idx)

    return loss_re, loss_kl

def evaluate(epoch, vae, test_loader, writer, device):
    if epoch % 10 != 0:
        pass
    Embedding = True

    sample_v = torch.randn(256, config.z_size).view(-1, config.z_size, 1, 1)

    with torch.no_grad():
        vae.eval()

        for batch_idx, (x, target) in enumerate(test_loader):
            # Embedding from testdata
            x = x.to(device)
            mu, logvar = vae.encode(x)
            mu, logvar = mu.squeeze(), logvar.squeeze()
            z = vae.reparameterize(mu, logvar)

            x = (x - x.min()) / (x.max() - x.min())
            z = (z - z.min()) / (z.max() - z.min())

            logger.debug(x.shape) # [64, 1, 64, 64]
            logger.debug(z.shape) # [64, 64]

            if Embedding:
                meta = []
                for i in range(x.shape[0]):
                    meta.append(str(i))
                writer.add_embedding(mat=z, metadata=meta, label_img=x)

            else:

                resultsample = torch.cat([x, z])
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

def main(args, device):
    logger.info("Begin train")
    train_loader, test_loader = load_processed_datasets(args.traindir)

    # start tensorboard
    writer = SummaryWriter(log_dir="../log/tensorboard/" + args.logdir, comment=args.comment)

    logger.info("define model")
    vae = VAE(zsize=config.z_size, layer_count=config.layer_count, channels=1)
    print(vae)
    vae.to(device)

    optimizer = optim.SGD(vae.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(1, args.epoch + 1):
        
        logger.info("Epoch: %d/%d" % (epoch, args.epoch))
        loss_re, loss_kl = train(epoch, vae, train_loader, optimizer, writer, device)
        logger.info("End epoch train")

        evaluate(epoch, vae, test_loader, writer, device)
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
    parse.add_argument("--logdir", type=str, default="default", help="set path of logfile ../log/tensorboard/[logdir]")
    parse.add_argument("--gpu_id", type=str, default="0",
                help="When you want to use 1 GPU, input 0. Using Multiple GPU, input [0, 1]")
    parse.add_argument("--traindir", type=str, default="processed", help="set path of train data dir ../../data/[traindir]")
    args = parse.parse_args()

    device = torch.device("cuda:" + args.gpu_id if torch.cuda.is_available() else "cpu")

    main(args, device)
