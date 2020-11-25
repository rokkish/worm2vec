"""With Redefine model, Analyze it in order to access output of the middle layer of.
"""
import torch
import torch.nn as nn
import train
import config
import get_logger
from my_args import args
from analyzer import Analyzer
from features.worm_dataset import WormDataset
import get_logger
logger = get_logger.get_logger(name='redefine_model')
device = torch.device("cuda:" + args.gpu_id if torch.cuda.is_available() else "cpu")

# 可視化
from tensorboardX import SummaryWriter


def main():
    logger.info("Begin analyze")

    _, test_loader = train.load_processed_datasets(
            args.traindir,
            window=0,
            sequential=args.sequential,
            shuffle={"train":True, "test":args.test_shuffle}
        )
    del _

    # start tensorboard
    writer = SummaryWriter(log_dir="../log/tensorboard/analyze_" + args.logdir)

    # Load model
    model = train.get_model(args.model)
    model.load_state_dict(torch.load("../models/" + args.model_name + ".pkl"))
    model.to(device)

    analyzer_ = Analyzer(
                            model,
                            writer,
                            device,
                            args.gpu_id,
                            args.max_analyze
                        )

    analyzer_.analyze(test_loader)

    # end tensorboard
    writer.close()
    logger.info("End analyze")


if __name__ == "__main__":

    logger.debug(args)
    main()
