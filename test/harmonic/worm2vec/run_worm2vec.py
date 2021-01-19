"""Train model, and evaluate it
"""
import os
import hydra
import wandb
import tensorflow as tf
import numpy as np
from omegaconf import DictConfig
from models.worm_model import deep_worm
from models.losses import proxy_anchor_loss, cosine_similarity_pos_neg, euclidian_distance_pos_neg
from trainer import Trainer
from predictor import Predictor
import get_logger
logger = get_logger.get_logger(name='run')


def get_binaryfile_number(val):
    """
        Args:
            val (path fo bmp file)  | ../../data/processed/alldata/date_000000.pt
        Return:
            file_number (int)       | 0
    """
    file_numbers = val.split("/")[-1].split(".pt")[0].split("_")
    file_number = file_numbers[-1]
    return int(file_number)

def np_load(path):
    """Check path and Return data(np.array)
    """
    if not os.path.exists(path):
        logger.debug(os.getcwd())
        raise ValueError("no exist")
    return np.load(path)["arr_0"]


def load_fixedtestdata(path):
    """When multi_run.py is running, test data must be fixed.
    """
    return np_load(path)


def load_fixedtestdata_forpredict(path):
    """must use minibatcher_withlabel
    """
    import glob
    return glob.glob(path+"/*.pt")


def load_data(params):
    # Load dataset (N, rot+neg, 1, H, W)

    if params.train_mode:
        test = load_fixedtestdata(params.path.fixedtestdata)
        dataset = np_load(params.path.worm_data)
    else:
        import torch
        test = load_fixedtestdata_forpredict(params.path.fixedtestdata)
        test.sort(key=get_binaryfile_number)
        dataset = np.zeros([10**4] + list(torch.load(test[0]).numpy().shape))

    # Split
    N = dataset.shape[0]
    valid_rate = params.preprocess.valid_rate
    train_rate = 1. - valid_rate
    N_tr = int(N * train_rate)

    train = dataset[:N_tr]
    valid = dataset[N_tr:]

    # Format
    data = {}
    data['train_x'] = train#[:1000]
    data['valid_x'] = valid#[:1000]
    data['test_x'] = test#[:1000]
    return data


def set_placeholders(batch_size, dim, n_positive, n_negative):
    with tf.name_scope('inputs'):
        positive = tf.placeholder(tf.float32,
                                [n_positive, dim, dim],
                                name='positive')
        negative = tf.placeholder(tf.float32,
                                [n_negative, dim, dim],
                                name='negative')
        learning_rate = tf.placeholder(tf.float32,
                                    name='learning_rate')
        train_phase = tf.placeholder(tf.bool, name='train_phase')
        class_id = tf.placeholder(tf.int32, name='class_id')
    return {"positive": positive,
            "negative": negative,
            "learning_rate": learning_rate,
            "train_phase": train_phase,
            "class_id": class_id}


def construct_model(params, placeholders):
    preds = {}
    n_sample = params.n_positive + params.n_negative

    ret = deep_worm(params,
                    pos=placeholders["positive"],
                    neg=placeholders["negative"],
                    train_phase=placeholders["train_phase"],
                    n_sample=n_sample,
                    reuse=None)
    with tf.name_scope('positive_embedding'):
        preds["positive"] = ret[:params.n_positive]
    with tf.name_scope('negative_embedding'):
        preds["negative"] = ret[params.n_positive:]
    return preds


def construct_loss(preds, class_id, params, sample_size):
    if params.nn.batch_size != 1:
        assert ValueError("batchsize must be 1. If not, calculating Proxy-anchor-loss is wrong.")

    return proxy_anchor_loss(
            embeddings=preds,
            class_id=class_id,
            n_classes=sample_size,
            n_unique=params.nn.n_positive,
            input_dim=params.nn.n_classes,
            alpha=params.loss.alpha,
            delta=params.loss.delta)


def construct_validloss(preds):
    """construct loss NOT for train.
    """
    return {"cossim": cosine_similarity_pos_neg(embeddings=preds),
            "eucliddist": euclidian_distance_pos_neg(embeddings=preds)}


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
    wandb.login()
    wandb.init(project="worm2vec-harmonic", name=cfg.exp_name)

    tf.reset_default_graph()
    logger.debug(cfg)

    # load_data
    data = load_data(cfg)

    # build model
    placeholders = set_placeholders(cfg.nn.batch_size, cfg.nn.dim, cfg.nn.n_positive, cfg.nn.n_negative)

    preds = construct_model(cfg.nn, placeholders)

    loss = construct_loss(preds, placeholders["class_id"], cfg, data["train_x"].shape[0])
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
        predictor = Predictor(cfg, loss, valid_loss, optim, train_op, placeholders)
        predictor.fit(data)


if __name__ == "__main__":
    main()
