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
from models.cboi import CBOI
from features.worm_dataset import WormDataset
from features.worm_transform import ToBinary, FillHole, Labelling, Padding, ToNDarray
from visualization.save_images_gray_grid import save_images_grid
import config
import get_logger
logger = get_logger.get_logger(name='train')

# 可視化
from tensorboardX import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter
#import matplotlib.pyplot as plt

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

def train(epoch, vae, train_loader, optimizer, writer, device, args):

    def loss_function(recon_x, x, mu, logvar):
        BCE = torch.mean((recon_x - x)**2)
        KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
        return BCE, KLD * 0.1

    def rotation_invariance_regularization(target, vae, z, rotation_invariant_rate=args.rotation_invariant_rate):

        def cut_z(z, rotation_invariant_rate=rotation_invariant_rate):
            return z[:, :int(z.shape[1]*rotation_invariant_rate)]

        RI = 0
        z = cut_z(z)

        for theta in range(target.shape[1]):
            mu, logvar = vae.encode(target[:, theta])
            mu, logvar = mu.squeeze(), logvar.squeeze()
            z_theta = vae.reparameterize(mu, logvar)
            z_theta = cut_z(z_theta)
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

        if args.use_rotate:
            loss_ri = rotation_invariance_regularization(target, vae, z)
            (loss_re + loss_kl + loss_ri).backward()
        else:
            #logger.debug("backward")
            (loss_re + loss_kl).backward()

        optimizer.step()

        if batch_idx % (len(train_loader) // 10) == 0:
            if args.use_rotate:
                logger.debug("Train batch: [{:0=4}/{} ({:0=2.0f}%)]\tLoss_re: {:.5f} \tLoss_kl: {:.5f} \tLoss_ri: {:.5f}".format(
                        batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss_re.item(), loss_kl.item(), loss_ri.item()))
            else:
                logger.debug("Train batch: [{:0=4}/{} ({:0=2.0f}%)]\tLoss_re: {:.5f} \tLoss_kl: {:.5f}".format(
                        batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss_re.item(), loss_kl.item()))

        writer.add_scalar(tag="train_loss_step_batch/re", scalar_value=loss_re.item(), global_step=batch_idx)
        writer.add_scalar(tag="train_loss_step_batch/kl", scalar_value=loss_kl.item(), global_step=batch_idx)
        if args.use_rotate:
            writer.add_scalar(tag="train_loss_step_batch/ri", scalar_value=loss_ri.item(), global_step=batch_idx)

    writer.add_scalar(tag="train_loss_step_epoch/re", scalar_value=loss_re.item(), global_step=epoch)
    writer.add_scalar(tag="train_loss_step_epoch/kl", scalar_value=loss_kl.item(), global_step=epoch)
    if args.use_rotate:
        writer.add_scalar(tag="train_loss_step_epoch/ri", scalar_value=loss_ri.item(), global_step=epoch)

def evaluate(epoch, vae, test_loader, writer, device):
    if epoch % 10 != 0:
        pass
    Embedding = True

    sample_v = torch.randn(256, config.z_size).view(-1, config.z_size, 1, 1)

    with torch.no_grad():
        vae.eval()

        for batch_idx, (x, target) in enumerate(test_loader):
            if batch_idx > 5:
                break

            x = x.to(device)

            if Embedding:
                # Embedding from testdata
                mu, logvar = vae.encode(x)
                mu, logvar = mu.squeeze(), logvar.squeeze()
                z = vae.reparameterize(mu, logvar)

                x = (x - x.min()) / (x.max() - x.min())
                z = (z - z.min()) / (z.max() - z.min())

                meta = []
                for i in range(x.shape[0]):
                    meta.append(str(i))
                writer.add_embedding(mat=z, metadata=meta, label_img=x, global_step=epoch, tag="Bacth_{0:0=3}".format(batch_idx))

            # Reconstruction from testdata
            x_rec, _, _ = vae.forward(x)
            x_rec = (x_rec - x_rec.min()) / (x_rec.max() - x_rec.min())

            result_rec_from_data = torch.cat([x, x_rec])
            result_rec_from_data = result_rec_from_data.cpu()

            save_images_grid(result_rec_from_data, nrow=16, scale_each=True, global_step=epoch,
                        tag_img="Rec_from_data/BATCH_{0:0=3}".format(batch_idx), writer=writer)

            # Reconstruction from random (mu, var)
            sample_v = sample_v.to(device)
            x_rec = vae.decode(sample_v)

            result_from_noise = x_rec * 0.5 + 0.5
            result_from_noise = result_from_noise.cpu()

            save_images_grid(result_from_noise, nrow=16, scale_each=True, global_step=epoch,
                        tag_img="Rec_from_noise/BATCH_{0:0=3}".format(batch_idx), writer=writer)

    #sample_v = torch.randn(128, config.z_size).view(-1, 1, config.z_size, config.z_size)
    #if epoch == 1:
    #    writer.add_graph(vae, sample_v)

def main(args, device):
    logger.info("Begin train")
    train_loader, test_loader = load_processed_datasets(args.traindir)

    # start tensorboard
    writer = SummaryWriter(log_dir="../log/tensorboard/" + args.logdir, comment=args.comment)

    logger.debug("define model")
    model = CBOI()
    model.to(device)

    optimizer = optim.SGD(vae.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(1, args.epoch + 1):
        
        logger.info("Epoch: %d/%d \tGPU: %d" % (epoch, args.epoch, int(args.gpu_id)))
        train(epoch, vae, train_loader, optimizer, writer, device, args)
        logger.debug("End epoch train")

        evaluate(epoch, vae, test_loader, writer, device)
        logger.debug("End epoch evaluation")

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
    parse.add_argument("--use_rotate", action="store_true", help="if true, train with rotate data, rotate invariant loss")
    parse.add_argument("--rotation_invariant_rate", type=float, default=1.0, help="define the rate of Rotation Invariant between (z, z_phi)")
    args = parse.parse_args()

    device = torch.device("cuda:" + args.gpu_id if torch.cuda.is_available() else "cpu")

    main(args, device)
