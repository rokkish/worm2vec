"""Predictor class mainly visualize embedding space.
"""
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
        self.latent_dim = params.predicting.latent_dim
        self.layers_name = [
            "encoder/maxpool4:0",
            "reshape3/Reshape:0",
        ]
        self.n_embedding = params.predicting.n_embedding
        self.constant_idx = 0

    def fit(self, data):
        sess = self.init_session()
        saver = tf.train.Saver()

        saver.restore(sess, self.checkpoint_fullpath)

        medium_op = list(
            map(
                lambda tname: tf.get_default_graph().get_tensor_by_name(tname),
                self.layers_name)
            )

        batcher = self.minibatcher(data["test_x"], data["test_label"])

        # def zero vec
        cat_latent = np.zeros((self.n_embedding, self.latent_dim))
        cat_recon_img = np.zeros((self.n_embedding, self.size, self.size))
        cat_img = np.zeros((self.n_embedding, self.size, self.size))

        labels = []

        for i, (x, x_label) in enumerate(batcher):
            if i == self.n_embedding:
                break

            feed_dict = {
                self.x: x,
                self.learning_rate: self.lr
            }
            summary = sess.run([medium_op], feed_dict=feed_dict)

            # select layer output
            latent_summary_np = np.array(summary[0][0])
            summary_np = np.array(summary[0][-1])

            # cat several vectors (output of model)
            cat_latent[i: (i+1)] = np.reshape(latent_summary_np[self.constant_idx], (1, self.latent_dim))
            cat_recon_img[i: (i+1)] = summary_np[self.constant_idx]

            # cat several images
            cat_img[i: (i+1)] = x[self.constant_idx]

            labels.append(x_label)

        cat_images = np.concatenate([cat_img, cat_recon_img], axis=0)

        # make sprite image (labels)
        cat_images = (1 - cat_images) * 255.
        img = create_sprite_image(cat_images)
        save_sprite_image(img, path=self.logdir + "sprite.png")

        # tensorize
        variables = tf.Variable(cat_latent, trainable=False, name="embedding_latent")

        # make tensor
        df = pd.DataFrame(cat_latent).astype("float64")
        df.to_csv(self.logdir + "tensor.csv", header=False, index=False, sep="\t")

        # make metadata.tsv (labels)
        with open(self.logdir + "metadata.tsv", "w") as f:
            f.write("Index\tLabel\n")
            for index, label in enumerate(labels):
                for i, str_label in enumerate(["square_circle", "triangle_circle", "square_triangle"]):
                    if label == str_label:
                        id_label = i

                f.write("%d\t%d\n" % (index, id_label))

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

    def minibatcher(self, inputs, labels):
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

            x = inputs[idx: idx + bs]

            # path:/root/~/circle_square/*.npy
            # label is circle_square
            label = labels[idx].split("/")[-2]

            for j in range(x.shape[1]):
                yield np.reshape(x[:, j], (bs, size, size)), label
