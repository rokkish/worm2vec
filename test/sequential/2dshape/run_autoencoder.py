"""Train model, and evaluate it
"""
#import os
#if debug
#os.chdir("./test/sequential/2dshape")

import glob
import random
import hydra
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig
import models.autoencoder as model
from models.autoencoder import nn_euclid_loss
from trainer_autoencoder import Trainer
from predictor_autoencoder import Predictor
import get_logger
logger = get_logger.get_logger(name='run_ae')
import wandb
from wandb.keras import WandbCallback


def load_data(path, test_rate, n_samples):
    # Load dataset (N, Time, H, W)
    cwd = hydra.utils.get_original_cwd()

    train_rate = 1. - test_rate

    dataset = {
        "train": np.zeros((int(3*train_rate*n_samples), 11, 64, 64)),
        "test": np.zeros((int(3*test_rate*n_samples), 11, 64, 64)),
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
        N_tr = int(N * train_rate)

        dataset["train"][i*int(train_rate*n_samples): (i+1)*int(train_rate*n_samples)] = arr[:N_tr]
        dataset["test"][i*int(test_rate*n_samples): (i+1)*int(test_rate*n_samples)] = arr[N_tr:]
        dataset["train_label"].extend(files[:N_tr])
        dataset["test_label"].extend(files[N_tr:])

    # Format
    data = {}
    data['train_x'] = dataset["train"]#[:1000]
    data['test_x'] = dataset["test"]#[:1000]
    data['train_label'] = dataset["train_label"]#[:1000]
    data['test_label'] = dataset["test_label"]#[:1000]
    return data


def preprocess(data):

    def normalize(x):
        return 1 - x / 255.

    data["train_x"] = normalize(data["train_x"])
    data["test_x"] = normalize(data["test_x"])

    return data


def set_placeholders(size, batchsize):
    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, [batchsize, size, size], name='x')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    return {"x": x,
            "learning_rate": learning_rate}


def construct_model(placeholders, input_dim, multiply_dim):
    ae = model.AutoEncoder(placeholders["x"], input_dim, multiply_dim)
    preds = ae.nn()
    return preds


def construct_loss(preds, placeholders):
    loss = nn_euclid_loss(x=placeholders["x"], recon_x=preds[0])
    """
    loss = nn_loss(x=placeholders["x"],
                   recon_x=preds[0],
                   z_mean=preds[1],
                   z_log_var=preds[2])
    """
    return loss


def set_optimizer(learning_rate):
    return tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.01)
#    return tf.train.AdamOptimizer(learning_rate=learning_rate)


@hydra.main(config_path="./conf/config_autoencoder.yaml")
def main(cfg: DictConfig):
    wandb.login()
    wandb.init(project="autoencoder for pretrain", name=cfg.exp_name)

    tf.reset_default_graph()
    random.seed(cfg.training.seed)
    logger.debug(cfg)

    # load_data
    data = load_data(cfg.dir.data, cfg.training.test_rate, cfg.training.n_samples)
    logger.debug(len(data["train_x"]))

    # TODO:need to normalize
    data = preprocess(data)

    # build model
    placeholders = set_placeholders(cfg.training.size, cfg.training.batchsize)

    preds = construct_model(placeholders, cfg.training.dim, cfg.training.multiply_dim)

    loss = construct_loss(preds, placeholders)

    optim = set_optimizer(cfg.training.learning_rate)

    grads_and_vars = optim.compute_gradients(loss)

    train_op = optim.apply_gradients(grads_and_vars)

    # train or predict
    if cfg.training.train_mode:
        trainer = Trainer(cfg, loss, optim, train_op, placeholders)
        trainer.fit(data)
    else:
        predictor = Predictor(cfg, loss, optim, train_op, placeholders)
        predictor = predictor.fit(data)


if __name__ == "__main__":
    main()
