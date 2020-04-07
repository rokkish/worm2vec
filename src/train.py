"""
    Train VAE, save params.
"""
from __future__ import print_function

import os

import torch
import torch.optim as optim
from torchvision import transforms

from my_args import args

# 自作
from models.cboi import CBOI
from features.worm_dataset import WormDataset
from trainer import Trainer
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
        transform=None, window=window)

    test_set = WormDataset(root="../../data/"+train_dir, train=False,
        transform=None, window=window)

    """ Dataloader """
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config.BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=config.BATCH_SIZE, shuffle=True)

    return train_loader, test_loader

def main(device):
    logger.info("Begin train")
    train_loader, test_loader = load_processed_datasets(args.traindir, args.window)

    # start tensorboard
    writer = SummaryWriter(log_dir="../log/tensorboard/" + args.logdir)

    logger.debug("define model")
    model = CBOI()
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    trainer = Trainer(model, optimizer, writer, device, \
        args.epoch, args.window, args.gpu_id, args.use_rotate)
    trainer.fit(train_loader)

    # end tensorboard
    writer.close()

    # Save model
    torch.save(model.state_dict(), "../models/CBOImodel.pkl")

if __name__ == "__main__":

    device = torch.device("cuda:" + args.gpu_id if torch.cuda.is_available() else "cpu")

    main(device)
