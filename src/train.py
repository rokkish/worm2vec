"""
    Train VAE, save params.
"""
from __future__ import print_function

import torch
import torch.optim as optim

# 自作
import config
import get_logger
from my_args import args
from models.vae import VAE
from models.cboi import CBOI
from models.continuous_bag_of_worm import CBOW
from trainer import Trainer
from features.worm_dataset import WormDataset
logger = get_logger.get_logger(name='train')
device = torch.device("cuda:" + args.gpu_id if torch.cuda.is_available() else "cpu")

# 可視化
from tensorboardX import SummaryWriter


def load_processed_datasets(train_dir, window, sequential, shuffle={"train":True, "test":True}):
    """ Set dataset """
    train_set = WormDataset(root="../../data/"+train_dir, train=True,
                            transform=None, window=window, sequential=sequential)

    test_set = WormDataset(root="../../data/"+train_dir, train=False,
                           transform=None, window=window, sequential=sequential)

    """ Dataloader """
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config.BATCH_SIZE, shuffle=shuffle["train"])

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=config.BATCH_SIZE, shuffle=shuffle["test"])

    return train_loader, test_loader


def get_model(model):
    if model == "vae":
        return VAE(zsize=args.zsize, layer_count=config.layer_count, channels=1)
    elif model == "cbow":
        return CBOW(zsize=args.zsize, loss_function_name=args.loss_function_name)
    else:
        raise NameError(model + " not exist")


def main():
    logger.info("Begin train")
    train_loader, test_loader = load_processed_datasets(args.traindir,
                                args.window, args.sequential)

    # start tensorboard
    writer = SummaryWriter(log_dir="../log/tensorboard/" + args.logdir)

    model = get_model(args.model)
    #TODO:init_weight()
    model.to(device)

    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    trainer = Trainer(model, optimizer, writer, device,
                      args.epoch, args.gpu_id)
    trainer.fit(train_loader, test_loader)

    # end tensorboard
    writer.close()

    # Save model
    torch.save(model.state_dict(), "../models/" + args.model_name + ".pkl")

    logger.info("End train")


if __name__ == "__main__":

    logger.debug(args)
    main()
