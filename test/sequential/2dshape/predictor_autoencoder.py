"""Predictor class mainly visualize embedding space.
"""
import sys
import time
import tensorflow as tf
import numpy as np
import pandas as pd
from trainer_autoencoder import Trainer
from tensorboard.plugins import projector

from visualize.sprite_images import create_sprite_image, save_sprite_image
import get_logger
logger = get_logger.get_logger(name="predictor")


class Predictor(Trainer):
    def __init__(self,
                 params,
                 loss,
                 optim,
                 train_op,
                 placeholders):
        super().__init__(params,
                         loss,
                         optim,
                         train_op,
                         placeholders)
        self.checkpoint_fullpath = params.dir.checkpoint_fullpath
        self.logdir = params.dir.tensorboard
        self.layers_name = [
            "reshape3/Reshape:0",
        ]
        self.n_embedding = params.predicting.n_embedding
        self.constant_idx = 0

    def fit(self, data):
        saver, sess = self.init_session()

        saver.restore(sess, self.checkpoint_fullpath)

        medium_op = list(
            map(
                lambda tname: tf.get_default_graph().get_tensor_by_name(tname),
                self.layers_name)
            )

        batcher = self.minibatcher(data["test_x"])

        # def zero vec
        cat_recon_img = np.zeros((self.n_embedding, self.size, self.size))
        cat_img = np.zeros((self.n_embedding, self.size, self.size))

        for i, x in enumerate(batcher):
            if i == self.n_embedding:
                break

            feed_dict = {
                self.x: x,
                self.learning_rate: self.lr
            }
            summary = sess.run([medium_op], feed_dict=feed_dict)

            # select last layer output
            summary_np = np.array(summary[-1][0])

            # cat several vectors (output of model)
            cat_recon_img[i: (i+1)] = summary_np[self.constant_idx]

            # cat several images
            cat_img[i: (i+1)] = x[self.constant_idx]

        cat_images = np.concatenate([cat_img, cat_recon_img], axis=0)

        # make sprite image (labels)
        cat_images /= 255.
        save_sprite_image(create_sprite_image(cat_images), path=self.logdir + "sprite.png")

        tf.summary.FileWriter(self.logdir, sess.graph)

        sess.close()
