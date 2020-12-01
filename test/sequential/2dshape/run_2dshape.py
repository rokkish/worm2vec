"""Train model, and evaluate it
"""
import glob
import random
import hydra
import tensorflow as tf
from omegaconf import DictConfig
from models.twoDshape_model import nn, nn_loss
#from trainer import Trainer
#from predictor import Predictor
import get_logger
logger = get_logger.get_logger(name='run')


def load_data(path, test_rate):
    # Load dataset (N, Time, H, W)
    cwd = hydra.utils.get_original_cwd()
    dataset = {"train": [], "test": []}
    data_label = ["circle_square", "circle_triangle", "square_triangle"]
    for label in data_label:
        dataset[label] = glob.glob(cwd + "/" + path + "/" + label + "/*.npy")
        random.shuffle(dataset[label])
        # Split
        N = len(dataset[label])
        train_rate = 1. - test_rate
        N_tr = int(N * train_rate)

        dataset["train"].extend(dataset[label][:N_tr])
        dataset["test"].extend(dataset[label][N_tr:])

    # Format
    data = {}
    data['train_x'] = dataset["train"]#[:1000]
    data['test_x'] = dataset["test"]#[:1000]
    return data


def set_placeholders(dim, window):
    with tf.name_scope('inputs'):
        x_previous = tf.placeholder(tf.float32, [window, dim, dim], name='x_previous')
        x_now = tf.placeholder(tf.float32, [1, dim, dim], name='x_now')
        x_next = tf.placeholder(tf.float32, [window, dim, dim], name='x_next')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    return {"x_previous": x_previous,
            "x_now": x_now,
            "x_next": x_next,
            "learning_rate": learning_rate}


def construct_model(input_dim, output_dim, placeholders):
    preds = nn(placeholders["x_previous"],
               placeholders["x_next"],
               placeholders["x_now"],
               input_dim, output_dim)
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
    logger.debug(data["train_x"][:10])

    # build model
    placeholders = set_placeholders(cfg.training.dim, cfg.training.window)

    preds = construct_model(cfg.training.dim, cfg.training.output_dim, placeholders)

    loss = construct_loss(preds)

    optim = set_optimizer(cfg.training.learning_rate)

    grads_and_vars = optim.compute_gradients(loss)

    train_op = optim.apply_gradients(grads_and_vars)

    # train or predict
    if cfg.training.train_mode:
        pass
    else:
        pass


if __name__ == "__main__":
    main()
