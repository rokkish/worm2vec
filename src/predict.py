"""
    Train VAE, save params.
"""
from __future__ import print_function

import os

import torch
import torch.optim as optim
from torchvision import transforms

from my_args import args
device = torch.device("cuda:" + args.gpu_id if torch.cuda.is_available() else "cpu")

# 自作
import config
import get_logger
from models.vae import VAE
from models.cboi import CBOI
from trainer import Trainer
from features.worm_dataset import WormDataset
logger = get_logger.get_logger(name='predict')

# 可視化
from tensorboardX import SummaryWriter

def load_processed_datasets(train_dir, window):

    test_set = WormDataset(root="../../data/"+train_dir, train=False,
        transform=None, window=window)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=config.BATCH_SIZE, shuffle=True)

    return test_loader

def main():
    logger.info("Begin train")
    test_loader = load_processed_datasets(args.traindir, args.window)

    # start tensorboard
    writer = SummaryWriter(log_dir="../log/tensorboard/predict_" + args.logdir)

    # Load model
    model = VAE(zsize=config.z_size, layer_count=config.layer_count, channels=1)
    model.load_state_dict(torch.load("../models/" + args.model_name + ".pkl"))
    model.to(device)

    #TODO:delete
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    trainer = Trainer(model, optimizer, writer, device, \
        args.epoch, args.window, args.gpu_id, args.use_rotate)

    for batch_idx, data_dic in enumerate(test_loader):
        data_idx, data = trainer.get_data_from_dic(data_dic)
        if data_idx == config.error_idx:
            logger.debug("Skip this batch beacuse window can't load data")
            continue
        else:
            target, context = trainer.slice_data(args.use_rotate, data)
            target, context = target.to(device), context.to(device)
        trainer.predict(context, epoch=0, batch_idx=0)
        break

    # end tensorboard
    writer.close()


if __name__ == "__main__":

    main()
