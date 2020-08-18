"""Train model, and evaluate it
"""
import os
import hydra
import tensorflow as tf
from omegaconf import DictConfig
from models.worm_model import deep_worm, triplet_loss
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
    positive = tf.placeholder(tf.float32,
                              [batch_size, dim**2],
                              name='positive')
    negative = tf.placeholder(tf.float32,
                              [batch_size, dim**2],
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
    for input_key in ["x", "positive", "negative"]:
        preds[input_key] = deep_worm(params,
                                     placeholders[input_key],
                                     placeholders["train_phase"])
    return preds


def set_optimizer(params):
    return tf.train.AdamOptimizer(learning_rate=params.learning_rate)


def set_train_op(grads_and_vars, optim, params):
    modified_gvs = []
    # We precondition the phases, for faster descent, in the same way as biases
    for g, v in grads_and_vars:
        if 'psi' in v.name:
            g = params.phase_preconditioner*g
        modified_gvs.append((g, v))
    return optim.apply_gradients(modified_gvs)


@hydra.main(config_path="./conf/config.yaml")
def main(cfg: DictConfig):
    logger.info("Begin run")

    # load_data
    # build model
    placeholders = set_placeholders(cfg.nn.batch_size, cfg.nn.dim)
    preds = construct_model(cfg.nn, placeholders)
    loss = triplet_loss(preds, cfg.loss.margin)
    optim = set_optimizer(cfg.optimizer)
    grads_and_vars = optim.compute_gradients(loss)
    trian_op = set_train_op(grads_and_vars, optim, cfg.nn)
    # train
    # test

    logger.info("End run")


if __name__ == "__main__":
    main()
