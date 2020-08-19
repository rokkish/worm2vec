"""Train model, and evaluate it
"""
import os
import hydra
import tensorflow as tf
import numpy as np
from omegaconf import DictConfig
from models.worm_model import deep_worm, triplet_loss
from trainer import Trainer
import get_logger
logger = get_logger.get_logger(name='run')


def load_data(params):
    # Load dataset (N, rot+neg, 1, H, W)
    if not os.path.exists(params.path.worm_data):
        logger.debug(os.getcwd())
        raise ValueError("no exist")
    dataset = np.load(params.path.worm_data)["arr_0"]

    # Split
    N = dataset.shape[0]
    valid_rate = params.preprocess.valid_rate
    test_rate = params.preprocess.test_rate
    train_valid_rate = 1. - test_rate
    train_rate = 1. - valid_rate
    N_trval = int(N * train_valid_rate)
    N_tr = int(N_trval * train_rate)

    train_valid = dataset[:N_trval]
    train = train_valid[:N_tr]
    valid = train_valid[N_tr:]
    test = dataset[N_trval:]

    # Format
    data = {}
    data['train_x'] = train
    data['valid_x'] = valid
    data['test_x'] = test
    return data


def set_placeholders(batch_size, dim):
    x = tf.placeholder(tf.float32,
                       [batch_size, dim, dim], name='x')
    positive = tf.placeholder(tf.float32,
                              [batch_size, dim, dim],
                              name='positive')
    negative = tf.placeholder(tf.float32,
                              [batch_size, dim, dim],
                              name='negative')
    learning_rate = tf.placeholder(tf.float32,
                                   name='learning_rate')
    train_phase = tf.placeholder(tf.bool, name='train_phase')
    return {"x": x,
            "positive": positive,
            "negative": negative,
            "learning_rate": learning_rate,
            "train_phase": train_phase}


def construct_model(params, placeholders):
    preds = {}
    for input_key, reuse in [("x", False), ("positive", True), ("negative", True)]:
        preds[input_key] = deep_worm(params,
                                     placeholders[input_key],
                                     placeholders["train_phase"],
                                     reuse)
    return preds


def set_optimizer(params):
    return tf.train.AdamOptimizer(learning_rate=params.learning_rate)


def modify_gvs(grads_and_vars, params):
    modified_gvs = []
    # We precondition the phases, for faster descent, in the same way as biases
    for g, v in grads_and_vars:
        if 'psi' in v.name:
            g = params.phase_preconditioner*g
        modified_gvs.append((g, v))
    return modified_gvs


@hydra.main(config_path="./conf/config.yaml")
def main(cfg: DictConfig):

    tf.reset_default_graph()

    # load_data
    data = load_data(cfg)
    logger.debug("tr:{}, va:{}, te:{}".format(data["train_x"].shape, data["valid_x"].shape, data["test_x"].shape))
    # build model
    placeholders = set_placeholders(cfg.nn.batch_size, cfg.nn.dim)
    preds = construct_model(cfg.nn, placeholders)
    loss = triplet_loss(preds, cfg.loss.margin)
    optim = set_optimizer(cfg.optimizer)
    grads_and_vars = optim.compute_gradients(loss)
    modified_gvs = modify_gvs(grads_and_vars, cfg.nn)
    trian_op = optim.apply_gradients(modified_gvs)
    # train
    trainer = Trainer(cfg, loss, optim, trian_op, placeholders)
    trainer.fit(data)


if __name__ == "__main__":
    main()
