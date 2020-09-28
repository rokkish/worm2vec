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
        self.n_classes = params.nn.n_classes
        self.layers_name = [
            "block4_1/Mean:0",
            "block4/Mean:0",
        ]
        self.n_embedding = 10
        self.view_pos_and_neg = False
        self.view_pos = True # if True view only pos,

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
                                   self.batch_size,
                                   shuffle=True)

        cat_summary_np = np.zeros((self.n_positive * self.n_embedding,
                                   self.n_classes))
        cat_Pos = np.zeros((self.n_positive * self.n_embedding,
                            self.dim, self.dim))

        cat_neg_summary_np = np.zeros((self.n_negative * self.n_embedding,
                                       self.n_classes))
        cat_Neg = np.zeros((self.n_negative * self.n_embedding,
                            self.dim, self.dim))

        for i, (Pos, Neg) in enumerate(batcher):
            if i == self.n_embedding:
                break

            feed_dict = {self.positive: Pos,
                         self.negative: Neg,
                         self.learning_rate: self.lr,
                         self.train_phase: False}
            summary = sess.run(medium_op, feed_dict=feed_dict)

            # select last layer output
            summary_np = np.array(summary[-1])
            summary_np = np.reshape(summary_np, (summary_np.shape[0], -1))

            # cat batch_embedding
            cat_summary_np[i*self.n_positive: (i+1)*self.n_positive] = summary_np
            cat_Pos[i*self.n_positive: (i+1)*self.n_positive] = Pos

            # select  first layer output
            summary_np = np.array(summary[0])
            summary_np = np.reshape(summary_np, (summary_np.shape[0], -1))

            # cat batch_embedding
            cat_neg_summary_np[i*self.n_negative: (i+1)*self.n_negative] = summary_np
            cat_Neg[i*self.n_negative: (i+1)*self.n_negative] = Neg

        if self.view_pos_and_neg:
            cat = np.concatenate([cat_summary_np, cat_neg_summary_np], axis=0)
            cat_img = np.concatenate([cat_Pos, cat_Neg], axis=0)
        else:
            if self.view_pos:
                cat = cat_summary_np
                cat_img = cat_Pos
            else:
                cat = cat_neg_summary_np
                cat_img = cat_Neg

        variables = tf.Variable(cat, trainable=False, name="embedding_lastlayer")

        # make tensor.tsv
        df = pd.DataFrame(cat).astype("float64")
        df.to_csv(self.logdir + "tensor.csv", header=False, index=False, sep="\t")

        # make metadata.tsv (labels)
        with open(self.logdir + "metadata.tsv", "w") as f:
            f.write("Index\tLabel\n")
            for index, label in enumerate(cat_img):
                # label means ID of image.
                if index >= self.n_positive * self.n_embedding:
                    custmom_label = self.n_positive + (index - self.n_positive * self.n_embedding) // self.n_negative
                else:
                    custmom_label = index//self.n_positive
                f.write("%d\t%d\n" % (index, custmom_label))

        # make sprite image (labels)
        save_sprite_image(create_sprite_image(cat_img), path=self.logdir + "sprite.png")

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
