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
        self.checkpoint_fullpath = params.dir.checkpoint_fullpath
        self.logdir = params.dir.tensorboard
        self.layers_name = [
            "concat_outputs/concat:0",
        ]
        self.n_embedding = params.predicting.n_embedding

    def fit(self, data):
        saver, sess = self.init_session()

        saver.restore(sess, self.checkpoint_fullpath)

        medium_op = list(
            map(
                lambda tname: tf.get_default_graph().get_tensor_by_name(tname),
                self.layers_name)
            )

        batcher = self.minibatcher(data["test_x"], shuffle=True)

        # def zero vec
        cat_context_summary_np = np.zeros((self.n_embedding, 64))
        cat_target_summary_np = np.zeros((self.n_embedding, 64))
        cat_context_img = np.zeros((self.n_embedding, self.dim, self.dim))
        cat_target_img = np.zeros((self.n_embedding, self.dim, self.dim))

        for i, (x_previous, x_now, x_next) in enumerate(batcher):
            if i == self.n_embedding:
                break

            feed_dict = {
                self.x_previous: x_previous,
                self.x_now: x_now,
                self.x_next: x_next,
                self.learning_rate: self.lr
            }
            summary = sess.run([medium_op], feed_dict=feed_dict)

            # select last layer output
            summary_np = np.array(summary[-1][0])

            # cat several vectors (output of model)
            cat_context_summary_np[i: (i+1)] = summary_np[0]
            cat_target_summary_np[i: (i+1)] = summary_np[1]

            # cat several images
            cat_context_img[i: (i+1)] = x_now
            cat_target_img[i: (i+1)] = x_now

        cat_tensors = np.concatenate([cat_context_summary_np, cat_target_summary_np], axis=0)
        cat_images = np.concatenate([cat_context_img, cat_target_img], axis=0)

        # tensorize
        variables = tf.Variable(cat_tensors, trainable=False, name="embedding_lastlayer")

        # make tensor.tsv
        df = pd.DataFrame(cat_tensors).astype("float64")
        df.to_csv(self.logdir + "tensor.csv", header=False, index=False, sep="\t")

        # make metadata.tsv (labels)
        with open(self.logdir + "metadata.tsv", "w") as f:
            f.write("Index\tLabel\n")
            for index, label in enumerate(cat_images):
                #TODO:cutom_label should be pair-name of circle, triangle, square.
                custmom_label = index
                f.write("%d\t%d\n" % (index, custmom_label))

        # make sprite image (labels)
        save_sprite_image(create_sprite_image(cat_images), path=self.logdir + "sprite.png")

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
