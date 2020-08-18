"""Train model, and evaluate it
"""
import os
import hydra
from omegaconf import DictConfig
import get_logger
logger = get_logger.get_logger(name='run')


def add_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        print('Created {:s}'.format(folder_name))
    return folder_name


@hydra.main(config_path="./conf/config.yaml")
def main(cfg: DictConfig):
    logger.info("Begin run")

    # load_data
    # build model
    # train
    # test

    logger.info("End run")


if __name__ == "__main__":
    main()
