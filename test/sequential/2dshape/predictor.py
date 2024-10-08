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
            "flatten/Reshape:0",
            "flatten/Reshape_1:0",
            "concat_outputs/concat:0",
        ]

        #FIXME:n_embedding is relative to batchsize & n_samples.
        self.batchsize = 1
        self.dataset_name = params.dir.data.split("/")[-1]
        if self.dataset_name == "morph":
            if params.training.n_samples % 10 != 0:
                raise ValueError("plot morph, n_samples%10 must be 0")
            self.n_embedding = int(params.training.n_samples-1)*3

        elif self.dataset_name == "minimorph":
            if params.training.n_samples != 240:
                raise ValueError("plot morph, n_samples must be 240")
            self.n_embedding = int(params.training.n_samples*0.3)

        self.constant_idx = 0
        self.output_dim = params.predicting.dim_out

    def restore_session(self):
        """Restore tensorflow Session

        Returns:
            saver: load self.config
            sess: load inital config
        """
        sess = tf.Session(config=self.config)
        sess.run([self.init_global,
                  self.init_local],
                 feed_dict={})

        subset_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="w_c") + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="w_enc")
        logger.debug(subset_variables)

        saver = tf.train.Saver(subset_variables)
        saver.restore(sess, self.checkpoint_fullpath)

        return saver, sess

    def fit(self, data):
        saver, sess = self.restore_session()

        medium_op = list(
            map(
                lambda tname: tf.get_default_graph().get_tensor_by_name(tname),
                self.layers_name)
            )

        batcher = self.minibatcher(data["test_x"], data["test_label"], shuffle=True)

        # def zero vec
        cat_context_summary_np = np.zeros((self.n_embedding, self.output_dim))
        cat_target_summary_np = np.zeros((self.n_embedding, self.output_dim))
        cat_shapetarget_summary_np = np.zeros((self.n_embedding, self.output_dim))
        cat_context_img = np.zeros((self.n_embedding, self.size, self.size))
        cat_target_img = np.zeros((self.n_embedding, self.size, self.size))
        cat_shapetarget_img = np.zeros((self.n_embedding, self.size, self.size))

        labels = []

        for i, (x_previous, x_now, x_next, x_label) in enumerate(batcher):
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
            summary_np = np.array(summary[0][-1])
            # select first layer output
            summary_np2 = np.array(summary[0][0])

            # cat several vectors (output of model)
            cat_context_summary_np[i: (i+1)] = summary_np[self.constant_idx]
            cat_target_summary_np[i: (i+1)] = summary_np[2*self.batchsize + self.constant_idx]
            cat_shapetarget_summary_np[i: (i+1)] = summary_np2[self.constant_idx]

            # cat several images
            cat_context_img[i: (i+1)] = x_now[self.constant_idx]
            cat_target_img[i: (i+1)] = x_now[self.constant_idx]
            cat_shapetarget_img[i: (i+1)] = x_now[self.constant_idx]

            labels.append(x_label)

        cat_tensors = np.concatenate([cat_context_summary_np, cat_target_summary_np, cat_shapetarget_summary_np], axis=0)
        cat_images = np.concatenate([cat_context_img, cat_target_img, cat_shapetarget_img], axis=0)
        cat_labels = labels + labels + labels
        # tensorize
        variables = tf.Variable(cat_tensors, trainable=False, name="embedding_lastlayer")

        # make tensor.tsv
        df = pd.DataFrame(cat_tensors).astype("float64")
        df.to_csv(self.logdir + "tensor.csv", header=False, index=False, sep="\t")

        # make metadata.tsv (labels)
        with open(self.logdir + "metadata.tsv", "w") as f:
            f.write("Index\tLabel\tContextOrTargetOrShapeTarget\n")
            for index, label in enumerate(cat_labels):
                for i, str_label in enumerate(["square_circle", "triangle_circle", "square_triangle"]):
                    if label == str_label:
                        id_label = i

                if index < cat_context_summary_np.shape[0]:
                    context_or_target_label = 0
                elif index >= cat_context_summary_np.shape[0] and index < 2*cat_context_summary_np.shape[0]:
                    context_or_target_label = 1
                else:
                    context_or_target_label = 2

                f.write("%d\t%d\t%d\n" % (index, id_label, context_or_target_label))

        # make sprite image (labels)
        cat_images = (1 - cat_images) * 255.
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
        embedding.sprite.single_image_dim.extend([self.size, self.size])

        summary_writer = tf.summary.FileWriter(self.logdir, sess.graph)
        projector.visualize_embeddings(summary_writer, config_projector)

        sess.close()

    def minibatcher(self, inputs, labels, shuffle=False):
        """

        Args:
            inputs (list): list of data path. (*.npy)
            shuffle (bool, optional): shffule idx. Defaults to False.

        Yields:
            x_previous, x_now, x_next, label (ndarray):
        """
        size = self.size
        bs = self.batchsize

        for idx in range(0, len(inputs), bs):

            # path:/root/~/circle_square/*.npy
            # label is circle_square
            x = inputs[idx: idx + bs]

            #TODO:pathを渡す
            label = labels[idx].split("/")[-2]

            if self.dataset_name == "morph":

                for m in range(0, x.shape[1]-2):
                    x_previous = np.reshape(x[:, 0+m], (bs, size, size))
                    x_now = np.reshape(x[:, 1+m], (bs, size, size))
                    x_next = np.reshape(x[:, 2+m], (bs, size, size))

                    yield x_previous, x_now, x_next, label

            elif self.dataset_name == "minimorph":
                x_previous = np.reshape(x[:, 0], (bs, size, size))
                x_now = np.reshape(x[:, 1], (bs, size, size))
                x_next = np.reshape(x[:, 2], (bs, size, size))

                yield x_previous, x_now, x_next, label
