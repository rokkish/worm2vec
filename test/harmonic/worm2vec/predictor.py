"""Predictor class mainly visualize embedding space.
"""
import sys
import time
import torch
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
                 valid_loss,
                 optim,
                 train_op,
                 placeholders):
        super().__init__(params,
                         loss,
                         valid_loss,
                         optim,
                         train_op,
                         placeholders)
        self.dim = params.nn.dim
        self.checkpoint_fullpath = params.path.checkpoint_fullpath
        self.logdir = params.path.tensorboard
        self.n_classes = params.nn.n_classes
        self.layers_name = [
            "block4/Mean:0",
            #"FCN/Relu:0"
        ]
        self.n_embedding = params.predict.n_embedding
        self.view_anchor = params.predict.view_anchor
        self.view_pos = params.predict.view_pos
        self.view_neg = params.predict.view_neg

        self.target_idx = params.predict.target_idx

        self.cossim_path_name = params.path.save_vector_path + "cossim_pn.csv"
        self.tensor_path_name = params.path.save_vector_path + "tensor.csv"
        self.metadata_path_name = params.path.save_vector_path + "metadata.tsv"
        self.sprite_image_path_name = self.logdir + "sprite.png"

        self.sprite_img_isSaved = params.predict.sprite_img_isSaved

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

        del data["train_x"]

        batcher = self.minibatcher_withlabel(data['test_x'])

        if self.view_anchor:
            cat_anchor_summary_np = np.zeros((self.n_embedding, self.n_classes))
            cat_Anchor = np.zeros((self.n_embedding, self.dim, self.dim))

        if self.view_pos:
            cat_summary_np = np.zeros((self.n_positive * self.n_embedding,
                                    self.n_classes))
            cat_Pos = np.zeros((self.n_positive * self.n_embedding,
                                self.dim, self.dim))

        if self.view_neg:
            cat_neg_summary_np = np.zeros((self.n_negative * self.n_embedding,
                                        self.n_classes))
            cat_Neg = np.zeros((self.n_negative * self.n_embedding,
                                self.dim, self.dim))

        cossim_pn = []
        labels = {"date":[], "id":[]}
        init_t = time.time()

        for i, (Pos, Neg, label_date, label_id) in enumerate(batcher):
            if i == self.n_embedding:
                break

            feed_dict = {self.positive: Pos,
                         self.negative: Neg,
                         self.learning_rate: self.lr,
                         self.train_phase: False}
            summary, cossim = sess.run([medium_op, self.cossim], feed_dict=feed_dict)

            # select last layer output
            summary_np = np.array(summary[-1])
            summary_np = np.reshape(summary_np, (summary_np.shape[0], -1))

            if self.view_anchor:
                # cat batch_embedding
                cat_anchor_summary_np[i] = summary_np[0]
                if self.sprite_img_isSaved:
                    cat_Anchor[i] = Pos[0]

            if self.view_pos:
                # cat batch_embedding
                cat_summary_np[i*self.n_positive: (i+1)*self.n_positive] = summary_np[:self.n_positive]
                if self.sprite_img_isSaved:
                    cat_Pos[i*self.n_positive: (i+1)*self.n_positive] = Pos

            if self.view_neg:
                # cat batch_embedding
                cat_neg_summary_np[i*self.n_negative: (i+1)*self.n_negative] = summary_np[self.n_positive:]
                if self.sprite_img_isSaved:
                    cat_Neg[i*self.n_negative: (i+1)*self.n_negative] = Neg

            cossim_pn.append(cossim["pn"])
            labels["date"].append(label_date)
            labels["id"].append(label_id)

            ptime = time.time() - init_t
            print("\r{:0=6}/{:0=6} ({:0=3.1f}%)[{:.1f} sec]".format(i, self.n_embedding, 100 * (i+1)/self.n_embedding, ptime), end="")

        self.create_cossim_csv(cossim_pn, self.cossim_path_name)

        cat_labels = {
            "date": labels["date"],
            "id": labels["id"]
            }

        if self.view_anchor:
            cat = cat_anchor_summary_np
            cat_img = cat_Anchor
        elif self.view_pos and self.view_neg and not self.view_anchor:
            cat = np.concatenate([cat_summary_np, cat_neg_summary_np], axis=0)
            cat_img = np.concatenate([cat_Pos, cat_Neg], axis=0)
            cat_labels = {
                "date": labels["date"] + labels["date"],
                "id": labels["id"] + labels["id"]
                }
        else:
            if self.view_pos and not self.view_neg:
                cat = cat_summary_np
                cat_img = cat_Pos
            elif self.view_neg and not self.view_pos:
                cat = cat_neg_summary_np
                cat_img = cat_Neg
            else:
                raise ValueError("setting of view_ is unexpected.")

        variables = tf.Variable(cat, trainable=False, name="embedding_lastlayer")

        self.create_tensor_tsv(cat, path=self.tensor_path_name)

        self.create_metadata_csv(cat_labels, path=self.metadata_path_name)

        self.mk_spriteimage(cat_img, True, path=self.sprite_image_path_name)

        config_projector = self.set_config_projector(variables)
        summary_writer = tf.summary.FileWriter(self.logdir, sess.graph)
        projector.visualize_embeddings(summary_writer, config_projector)

        sess.close()

    def minibatcher_withlabel(self, inputs, shuffle=False):
        """
        Args:
            inputs (list): list of data path.
        Yields:
            positive: positive data
            negative: negative data
            class_id: id of positive proxy
            date: yymmddHHMMSS
            id_: 000000 ~ 999999
        """

        indices = np.arange(len(inputs))
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(len(inputs)):
            excerpt = indices[start_idx]
            path = inputs[excerpt]
            arr = torch.load(path).numpy()
            labels = path.split("/")[-1].split(".pt")[0].split("_")

            date, id_ = labels[0], labels[-1]
            #if len(labels) != 2:
            #raise ValueError("len(labels) is not 2, but {}".format(len(labels)))

            yield arr[:self.n_positive, 0], \
                arr[-self.n_negative:, 0], \
                date, id_

    def create_cossim_csv(self, cossim_pn, path):
        # save predictor cossim.csv
        pd.DataFrame(cossim_pn).to_csv(path)

    def create_tensor_tsv(self, cat_tensors, path):
        # make tensor.tsv
        df = pd.DataFrame(cat_tensors).astype("float64")
        df.to_csv(path, header=False, index=False, sep="\t")

    def create_metadata_csv(self, cat_labels, path):
        # make metadata.tsv (labels)
        with open(path, "w") as f:
            f.write("Index\tLabel_date\tLabel_id\tAnchor_Pos_Neg\n")
            for index in range(len(cat_labels["date"])):
                date_ = int(cat_labels["date"][index])
                id_ = int(cat_labels["id"][index])
                anc_pos_neg = self.anc_pos_neg_idx(index)
                f.write("%d\t%d\t%d\t%d\n" % (index, date_, id_, anc_pos_neg))

    def mk_spriteimage(self, cat_images, isInversed, path):
        # make sprite image (labels)
        if self.sprite_img_isSaved:
            if isInversed:
                cat_images = (1 - cat_images) * 255.
            save_sprite_image(create_sprite_image(cat_images), path=path)

    def set_config_projector(self, variables):
        # config of projector
        config_projector = projector.ProjectorConfig()
        embedding = config_projector.embeddings.add()
        embedding.tensor_name = variables.name
        # tsv path
        embedding.tensor_path = self.tensor_path_name
        embedding.metadata_path = self.metadata_path_name
        # config of sprite image
        embedding.sprite.image_path = self.sprite_image_path_name
        embedding.sprite.single_image_dim.extend([self.dim, self.dim])
        return config_projector

    def anc_pos_neg_idx(self, idx):
        """
        Returns:
            anc_pos_neg: 0:anchor, 1:pos, 2:neg
        """
        if self.view_anchor:
            anc_pos_neg = 0
        elif self.view_pos and self.view_neg and not self.view_anchor:
            if idx < self.n_positive:
                anc_pos_neg = 1
            else:
                anc_pos_neg = 2
        else:
            raise ValueError("setting of view_ is unexpected.")

        return anc_pos_neg
