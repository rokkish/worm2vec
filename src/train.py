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
from trainer import Trainer
from features.worm_transform import ToBinary, FillHole, Labelling, Padding, ToNDarray
from visualization.save_images_gray_grid import save_images_grid
import config
import get_logger
logger = get_logger.get_logger(name='train')

# 可視化
from tensorboardX import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter
#import matplotlib.pyplot as plt

def load_processed_datasets(train_dir, window):
    """ Set dataset """
    train_set = WormDataset(root="../../data/"+train_dir, train=True,
        transform=None, processed=True, window=window)

    test_set = WormDataset(root="../../data/"+train_dir, train=False,
        transform=None, processed=True, window=window)

    """ Dataloader """
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config.BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=config.BATCH_SIZE, shuffle=True)

    return train_loader, test_loader

def main(args, device):
    logger.info("Begin train")
    train_loader, test_loader = load_processed_datasets(args.traindir, args.window)

    # start tensorboard
    writer = SummaryWriter(log_dir="../log/tensorboard/" + args.logdir)

    logger.debug("define model")
    model = CBOI()
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    trainer = Trainer(model, optimizer, writer, device)
    trainer.fit(train_loader, args)

    # end tensorboard
    writer.close()

    # Save model
    torch.save(model.state_dict(), "../models/CBOImodel.pkl")

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-e", "--epoch", type=int, default=15)
    parse.add_argument("--logdir", type=str, default="default", help="set path of logfile ../log/tensorboard/[logdir]")
    parse.add_argument("--gpu_id", type=str, default="0",
                help="When you want to use 1 GPU, input 0. Using Multiple GPU, input [0, 1]")
    parse.add_argument("--traindir", type=str, default="processed_split_date/201302081353", help="set path of train data dir ../../data/[traindir]")
    parse.add_argument("--window", type=int, default=3)
    args = parse.parse_args()

    device = torch.device("cuda:" + args.gpu_id if torch.cuda.is_available() else "cpu")

    main(args, device)
