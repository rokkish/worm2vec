"""Train model, and evaluate it
"""
import glob
import random
import hydra
import wandb
import numpy as np
import pandas as pd
import tensorflow as tf
from omegaconf import DictConfig
from models.worm2vec_skipgram import skipgram, skipgram_loss
from trainer import Trainer
from predictor import Predictor
import get_logger
logger = get_logger.get_logger(name='run')


def load_np(path, n_samples):
    tensor = pd.read_csv(path, header=None, sep="\t")
    if n_samples >= tensor.shape[0]:
        n_samples = None
    return tensor.values[:n_samples]


def load_labels(path, n_samples):
    """
    Returns:
        DataFrame:
            columns: 'Label_date', 'Label_id', 'Anchor_Pos_Neg'
            val: int, int, int
    """
    metadata = pd.read_csv(path, index_col=0, sep="\t")
    if n_samples >= metadata.shape[0]:
        n_samples = None
    return metadata.iloc[:n_samples]


def load_data(path, test_path, metadata_path, test_metadata_path, n_samples, test_n_samples):
    # Load dataset (N, 1, H, W)

    arr = load_np(path, n_samples)
    test_arr = load_np(test_path, test_n_samples)
    labels = load_labels(metadata_path, n_samples)
    test_labels = load_labels(test_metadata_path, test_n_samples)

    # Format
    data = {}
    data['train_x'] = arr
    data['test_x'] = test_arr
    data['train_label'] = labels
    data['test_label'] = test_labels
    return data


def set_placeholders(dim, batchsize):
    with tf.name_scope('inputs'):
        x_previous = tf.placeholder(tf.float32, [batchsize, dim], name='x_previous')
        x_now = tf.placeholder(tf.float32, [batchsize, dim], name='x_now')
        x_next = tf.placeholder(tf.float32, [batchsize, dim], name='x_next')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    return {"x_previous": x_previous,
            "x_now": x_now,
            "x_next": x_next,
            "learning_rate": learning_rate}


def construct_model(placeholders, input_dim):
    model = skipgram(placeholders["x_previous"],
                            placeholders["x_next"],
                            placeholders["x_now"],
                            input_dim)
    preds = model.nn()
    return preds


def construct_loss(preds, batchsize):
    loss = skipgram_loss(context=preds[:2*batchsize], target=preds[2*batchsize:])
    return loss


def set_optimizer(learning_rate, optim):
    if optim == "sgd":
        return tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.01)
    elif optim == "adam":
        return tf.train.AdamOptimizer(learning_rate=learning_rate)


@hydra.main(config_name="./conf/config")
def main(cfg: DictConfig):
    wandb.login()
    wandb.init(project="worm2vec-skipgram", name=cfg.exp_name)

    tf.reset_default_graph()
    random.seed(cfg.train.seed)
    logger.debug(cfg)

    # load_data
    data = load_data(cfg.dir.data, cfg.dir.test.data, cfg.dir.metadata, cfg.dir.test.metadata, cfg.train.n_samples, cfg.test.n_samples)

    # build model
    placeholders = set_placeholders(cfg.train.dim, cfg.train.batchsize)

    preds = construct_model(placeholders, cfg.train.dim)

    loss = construct_loss(preds, cfg.train.batchsize)

    optim = set_optimizer(cfg.train.learning_rate, cfg.train.optim)

    grads_and_vars = optim.compute_gradients(loss)

    train_op = optim.apply_gradients(grads_and_vars)

    # train or predict
    if cfg.train.train_mode:
        trainer = Trainer(cfg, loss, optim, train_op, placeholders)
        trainer.fit(data)
    else:
        predictor = Predictor(cfg, loss, optim, train_op, placeholders)
        predictor.fit(data)


if __name__ == "__main__":
    main()
