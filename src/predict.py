"""
    Train VAE, save params.
"""
from __future__ import print_function

import torch
import torch.optim as optim

from my_args import args

# 自作
import config
import get_logger
from models.vae import VAE
from models.cboi import CBOI
from predictor import Predictor
from features.worm_dataset import WormDataset
logger = get_logger.get_logger(name='predict')
device = torch.device("cuda:" + args.gpu_id if torch.cuda.is_available() else "cpu")
import train

# 可視化
from tensorboardX import SummaryWriter


def main():
    logger.info("Begin predict")
    train_loader, test_loader = train.load_processed_datasets(
            args.traindir,
            window=0,
            sequential=args.sequential,
            shuffle={"train":True, "test":args.test_shuffle}
        )
    loader = {"train":train_loader, "test":test_loader}

    # start tensorboard
    writer = SummaryWriter(log_dir="../log/tensorboard/predict_" + args.logdir)

    # Load model
    model = train.get_model(args.model)

    model.load_state_dict(torch.load("../models/" + args.model_name + ".pkl"))
    model.to(device)

    predictor = Predictor(
                            model,
                            writer,
                            device,
                            args.gpu_id,
                            args.max_predict
                        )

    predictor.predict(loader[args.loader])

    # end tensorboard
    writer.close()
    logger.info("End predict")


if __name__ == "__main__":

    logger.debug(args)
    main()
