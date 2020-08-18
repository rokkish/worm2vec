"""Train model, and evaluate it
"""
import os
import hydra
import tensorflow as tf
from omegaconf import DictConfig
import get_logger
logger = get_logger.get_logger(name='run')


def add_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        print('Created {:s}'.format(folder_name))
    return folder_name


def set_placeholders(batch_size, dim):
    x = tf.placeholder(tf.float32,
                       [batch_size, dim**2], name='x')
    learning_rate = tf.placeholder(tf.float32,
                                   name='learning_rate')
    train_phase = tf.placeholder(tf.bool, name='train_phase')
    return {"x": x,
            "learning_rate": learning_rate,
            "train_phase": train_phase}


@hydra.main(config_path="./conf/config.yaml")
def main(cfg: DictConfig):
    logger.info("Begin run")

    # load_data
    # build model
    placeholders = set_placeholders(cfg.nn.batch_size, cfg.nn.dim)
    # train
    # test

    logger.info("End run")


if __name__ == "__main__":
    main()
