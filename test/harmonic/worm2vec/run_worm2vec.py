"""Train model, and evaluate it
"""
import os
import hydra
import tensorflow as tf
import numpy as np
from omegaconf import DictConfig
from models.worm_model import deep_worm
from models.losses import triplet_loss, proxy_anchor_loss, cosine_similarity_pos_neg
from trainer import Trainer
from predictor import Predictor
import get_logger
logger = get_logger.get_logger(name='run')


def np_load(path):
    """Check path and Return data(np.array)
    """
    if not os.path.exists(path):
        logger.debug(os.getcwd())
        raise ValueError("no exist")
    return np.load(path)["arr_0"]


def load_fixedtestdata():
    """When multi_run.py is running, test data must be fixed.
    """
    return np_load("/root/worm2vec/data/variety_data_r36_n50_np/test/00.npz")


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

    train = dataset[:N_trval]
    valid = dataset[N_trval:]
    test = load_fixedtestdata()

    # Format
    data = {}
    data['train_x'] = train
    data['valid_x'] = valid
    data['test_x'] = test
    return data


def set_placeholders(batch_size, dim, n_positive, n_negative):
    positive = tf.placeholder(tf.float32,
                              [n_positive, dim, dim],
                              name='positive')
    negative = tf.placeholder(tf.float32,
                              [n_negative, dim, dim],
                              name='negative')
    learning_rate = tf.placeholder(tf.float32,
                                   name='learning_rate')
    train_phase = tf.placeholder(tf.bool, name='train_phase')
    return {"positive": positive,
            "negative": negative,
            "learning_rate": learning_rate,
            "train_phase": train_phase}


def construct_model(params, placeholders):
    preds = {}
    n_sample = {"positive": params.n_positive, "negative": params.n_negative}

    for input_key, reuse in [("positive", False), ("negative", True)]:
        preds[input_key] = deep_worm(params,
                                     placeholders[input_key],
                                     placeholders["train_phase"],
                                     n_sample[input_key],
                                     reuse)
    return preds


def construct_loss(preds, params, sample_size):
    if params.nn.batch_size != 1:
        assert ValueError("batchsize must be 1. If not, calculating Proxy-anchor-loss is wrong.")

    return proxy_anchor_loss(
            embeddings=preds,
            n_classes=sample_size,
            n_unique=params.nn.n_positive,
            input_dim=params.nn.n_classes,
            alpha=params.loss.alpha,
            delta=params.loss.delta)


def construct_validloss(preds):
    return cosine_similarity_pos_neg(embeddings=preds)


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
    placeholders = set_placeholders(cfg.nn.batch_size, cfg.nn.dim, cfg.nn.n_positive, cfg.nn.n_negative)
    preds = construct_model(cfg.nn, placeholders)
    loss = construct_loss(preds, cfg, data["train_x"].shape[0])
    valid_loss = construct_validloss(preds)
    optim = set_optimizer(cfg.optimizer)
    grads_and_vars = optim.compute_gradients(loss)
    modified_gvs = modify_gvs(grads_and_vars, cfg.nn)
    train_op = optim.apply_gradients(modified_gvs)
    # train or predict
    if cfg.train_mode:
        trainer = Trainer(cfg, loss, valid_loss, optim, train_op, placeholders)
        trainer.fit(data)
    else:
        predictor = Predictor(cfg, loss, optim, train_op, placeholders)
        predictor.fit(data)


if __name__ == "__main__":
    main()
