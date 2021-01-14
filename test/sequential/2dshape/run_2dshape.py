"""Train model, and evaluate it
"""
import glob
import random
import hydra
import wandb
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig
from models.twoDshape_model import TwoDshape_model, nn_loss
from models.twoDshapeAE_model import TwoDshapeAE_model, nn_ae_loss
from models.twoDshapeCBOW_model import TwoDshapeCBOW_model, nn_cbow_loss
from models.twoDshapeSKIPGRAM_model import TwoDshapeSKIPGRAM_model, nn_skipgram_loss
from trainer import Trainer
from predictor import Predictor
import get_logger
logger = get_logger.get_logger(name='run')


def dataset_dim(dataset_name):
    dataset_name = dataset_name.split("/")[-1]
    if dataset_name == "morph":
        return 11
    elif dataset_name == "minimorph":
        return 3
    else:
        raise ValueError("Unknown dataset")


def load_data(path, test_rate, n_samples, dataset_name=None):
    # Load dataset (N, Time, H, W)
    cwd = hydra.utils.get_original_cwd()

    train_rate = 1. - test_rate

    data_dim = dataset_dim(dataset_name)
    
    dataset = {
        "train": np.zeros((int(3*train_rate*n_samples), data_dim, 64, 64)),
        "test": np.zeros((int(3*test_rate*n_samples), data_dim, 64, 64)),
        "train_label": [],
        "test_label": []
    }

    data_label = ["square_circle", "triangle_circle", "square_triangle"]

    for i, label in enumerate(data_label):

        files = sorted(glob.glob(cwd + "/" + path + "/" + label + "/*.npy"))[:n_samples]
        random.shuffle(files)

        arr = np.zeros((len(files), data_dim, 64, 64))

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
        x_previous = tf.placeholder(tf.float32, [batchsize, size, size], name='x_previous')
        x_now = tf.placeholder(tf.float32, [batchsize, size, size], name='x_now')
        x_next = tf.placeholder(tf.float32, [batchsize, size, size], name='x_next')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    return {"x_previous": x_previous,
            "x_now": x_now,
            "x_next": x_next,
            "learning_rate": learning_rate}


def construct_model(placeholders, input_dim, multiply_dim, share_enc_trainable):
    model = TwoDshapeSKIPGRAM_model(placeholders["x_previous"],
                            placeholders["x_next"],
                            placeholders["x_now"],
                            input_dim,
                            multiply_dim,
                            share_enc_trainable)
    preds = model.nn()
    return preds


def construct_loss(preds, batchsize):
    loss = nn_skipgram_loss(context=preds[:2*batchsize], target=preds[2*batchsize:])
    return loss


def set_optimizer(learning_rate):
    return tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.01)


@hydra.main(config_path="./conf/config.yaml")
def main(cfg: DictConfig):
    wandb.login()
    wandb.init(project="2dshape-skipgram", name=cfg.exp_name)

    tf.reset_default_graph()
    random.seed(cfg.training.seed)
    logger.debug(cfg)

    # load_data
    data = load_data(cfg.dir.data, cfg.training.test_rate, cfg.training.n_samples, cfg.dir.data)
    logger.debug(len(data["train_x"]))

    # TODO:need to normalize
    data = preprocess(data)

    # build model
    placeholders = set_placeholders(cfg.training.size, cfg.training.batchsize)

    preds = construct_model(placeholders, cfg.training.dim, cfg.training.multiply_dim, cfg.training.share_enc_trainable)

    loss = construct_loss(preds, cfg.training.batchsize)

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
