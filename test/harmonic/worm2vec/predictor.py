"""Predictor class mainly visualize embedding space.
"""
import sys
import time
import tensorflow as tf
import numpy as np
import pandas as pd
from trainer import Trainer
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
        self.dim = params.nn.dim
        self.checkpoint_fullpath = params.path.checkpoint_fullpath
        self.logdir = params.path.tensorboard
        self.layers_name = [
            "block1/hconv1/Reshape_1:0",
            "block2/hconv3/Reshape_1:0",
            "block3/hconv5/Reshape_1:0",
            "block4/hconv7/Reshape_1:0",
            "block4/hconv7/Reshape:0",
            "block4/Mean:0",
            "block4/Maximum:0"
        ]

    def fit(self, data):
        saver = tf.train.Saver()
        sess = tf.Session(config=self.config)
        sess.run([self.init_global,
                  self.init_local],
                 feed_dict={self.train_phase: True})

        saver.restore(sess, self.checkpoint_fullpath)

        medium_op = list(
            map(
                lambda tname: tf.get_default_graph().get_tensor_by_name(tname),
                self.layers_name)
            )

        batcher = self.minibatcher(data['test_x'],
                                   self.batch_size)

        for i, (X, Pos, Neg) in enumerate(batcher):
            feed_dict = {self.x: X,
                         self.positive: Pos,
                         self.negative: Neg,
                         self.learning_rate: self.lr,
                         self.train_phase: False}
            summary = sess.run(medium_op, feed_dict=feed_dict)

            # select last layer output
            summary_np = np.array(summary[-1])
            summary_np = np.reshape(summary_np, (summary_np.shape[0], -1))
            variables = tf.Variable(summary_np, trainable=False, name="embedding_lastlayer")

            # make tensor.tsv
            df = pd.DataFrame(summary_np).astype("float64")
            df.to_csv(self.logdir + "tensor.csv", header=False, index=False, sep="\t")

            # make metadata.tsv (labels)
            with open(self.logdir + "metadata.tsv", "w") as f:
                f.write("Index\tLabel\n")
                for index, label in enumerate(X):
                    # FIXME:label dont exist in worm data.
                    f.write("%d\t%d\n" % (index, int(index)))

            # make sprite image (labels)
            # X = (Batch, H, W)
            # TODO:X, Pos, Negを一緒にEmbeddingして可視化する．
            save_sprite_image(create_sprite_image(X), path=self.logdir + "sprite.png")
            break

        # config of projector
        config_projector = projector.ProjectorConfig()
        embedding = config_projector.embeddings.add()
        embedding.tensor_name = variables.name
        # tsv path
        embedding.tensor_path = "tensor.csv"
        embedding.metadata_path = "metadata.tsv"
        # config of sprite image
        embedding.sprite.image_path = "sprite.png"
        embedding.sprite.single_image_dim.extend([self.dim, self.dim])

        summary_writer = tf.summary.FileWriter(self.logdir, sess.graph)
        projector.visualize_embeddings(summary_writer, config_projector)

        sess.close()
