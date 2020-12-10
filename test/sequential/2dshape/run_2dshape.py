"""Train model, and evaluate it
"""
import os
#if debug
#os.chdir("./test/sequential/2dshape")

import glob
import random
import hydra
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig
from models.twoDshape_model import nn, nn_loss
from trainer import Trainer
from predictor import Predictor
import get_logger
logger = get_logger.get_logger(name='run')


def load_data(path, test_rate):
    # Load dataset (N, Time, H, W)
    cwd = hydra.utils.get_original_cwd()

    n_samples = 20000

    dataset = {
        "train": np.zeros((int(3*0.9*n_samples), 11, 64, 64)),
        "test": np.zeros((int(3*0.1*n_samples), 11, 64, 64)),
        "train_label": [],
        "test_label": []
    }

    data_label = ["square_circle", "triangle_circle", "square_triangle"]

    for i, label in enumerate(data_label):

        files = sorted(glob.glob(cwd + "/" + path + "/" + label + "/*.npy"))[:n_samples]
        random.shuffle(files)

        arr = np.zeros((len(files), 11, 64, 64))

        for j, f in enumerate(files):
            arr[j] = np.load(f)
            print("\r Loading... {:0=5}/{:0=5}".format(j+i*n_samples, 3*len(files)), end="")

        # Split
        N = len(arr)
        train_rate = 1. - test_rate
        N_tr = int(N * train_rate)

        dataset["train"][i*int(0.9*n_samples): (i+1)*int(0.9*n_samples)] = arr[:N_tr]
        dataset["test"][i*int(0.1*n_samples): (i+1)*int(0.1*n_samples)] = arr[N_tr:]
        dataset["train_label"].extend(files[:N_tr])
        dataset["test_label"].extend(files[N_tr:])

    # Format
    data = {}
    data['train_x'] = dataset["train"]#[:1000]
    data['test_x'] = dataset["test"]#[:1000]
    data['train_label'] = dataset["train_label"]#[:1000]
    data['test_label'] = dataset["test_label"]#[:1000]
    return data


def set_placeholders(dim, batchsize):
    with tf.name_scope('inputs'):
        x_previous = tf.placeholder(tf.float32, [batchsize, dim, dim], name='x_previous')
        x_now = tf.placeholder(tf.float32, [batchsize, dim, dim], name='x_now')
        x_next = tf.placeholder(tf.float32, [batchsize, dim, dim], name='x_next')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    return {"x_previous": x_previous,
            "x_now": x_now,
            "x_next": x_next,
            "learning_rate": learning_rate}


def construct_model(input_dim, placeholders):
    preds = nn(placeholders["x_previous"],
               placeholders["x_next"],
               placeholders["x_now"],
               input_dim)
    return preds


def construct_loss(preds):
    loss = nn_loss(context=preds[0], target=preds[1])
    return loss


def set_optimizer(learning_rate):
    return tf.train.AdamOptimizer(learning_rate=learning_rate)


@hydra.main(config_path="./conf/config.yaml")
def main(cfg: DictConfig):

    tf.reset_default_graph()
    random.seed(cfg.training.seed)
    logger.debug(cfg)

    # load_data
    data = load_data(cfg.dir.data, cfg.training.test_rate)
    logger.debug(len(data["train_x"]))

    # build model
    placeholders = set_placeholders(cfg.training.dim, cfg.training.batchsize)

    preds = construct_model(cfg.training.dim, placeholders)

    loss = construct_loss(preds)

    optim = set_optimizer(cfg.training.learning_rate)

    grads_and_vars = optim.compute_gradients(loss)

    train_op = optim.apply_gradients(grads_and_vars)

    # train or predict
    if cfg.training.train_mode:
        trainer = Trainer(cfg, loss, optim, train_op, placeholders)
        trainer.fit(data)
    else:
        predictor = Predictor(cfg, loss, optim, train_op, placeholders)
        predictor.fit(data)


if __name__ == "__main__":
    main()
