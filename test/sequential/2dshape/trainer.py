"""Trainer class mainly train model
"""
import os
import time
import wandb
import tensorflow as tf
import numpy as np
import pandas as pd
import get_logger
logger = get_logger.get_logger(name="trainer")
from visualize.sprite_images import create_sprite_image, save_sprite_image
from tensorboard.plugins import projector


class Trainer(object):
    """class for training sequential model.
    """
    def __init__(self,
                 params,
                 loss,
                 optim,
                 train_op,
                 placeholders):

        # initialize
        self.loss = loss
        self.optim = optim
        self.train_op = train_op

        # yaml config
        self.lr = params.training.learning_rate
        self.n_epochs = params.training.n_epochs
        self.batchsize = params.training.batchsize
        self.checkpoint = params.dir.checkpoint
        self.csv_path = params.dir.losscsv
        self.dim = params.training.dim
        self.size = params.training.size
        self.restore = params.training.restore
        self.checkpoint_fullpath_subset = params.dir.checkpoint_fullpath_subset

        # placeholders
        self.x_previous = placeholders["x_previous"]
        self.x_now = placeholders["x_now"]
        self.x_next = placeholders["x_next"]
        self.learning_rate = placeholders["learning_rate"]

        # summary_writer
        self.train_summary_writer = tf.summary.FileWriter("./tensorboard/train")
        self.test_summary_writer = tf.summary.FileWriter("./tensorboard/test")

        # Configure tensorflow session
        self.init_global = tf.global_variables_initializer()
        self.init_local = tf.local_variables_initializer()
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = params.device.gpu.allow_growth
        self.config.gpu_options.visible_device_list = params.device.gpu.id
        self.config.log_device_placement = params.device.gpu.log_device_placement

        # morph or minimorph for switch minibatcher
        self.dataset_name = params.dir.data.split("/")[-1]

        ### predict ###
        self.default_logdir = params.dir.tensorboard
        self.layers_name = [
            "flatten/Reshape:0",
            "concat_outputs/concat:0"
        ]
        if self.dataset_name == "morph":
            self.n_embedding = 27
        elif self.dataset_name == "minimorph":
            self.n_embedding = int(params.training.n_samples*params.training.test_rate)//params.training.batchsize*3
        self.output_dim = 256
        self.constant_idx = 0
        self.label_dict = {
            "square_circle": 0,
            "triangle_circle": 1,
            "square_triangle": 2
        }
        self.tensor_path_name = "tensor.csv"
        self.metadata_path_name = "metadata.tsv"
        self.sprite_image_path_name = "sprite.png"

    def init_session(self):
        """Initial tensorflow Session

        Returns:
            sess: load inital config
        """
        sess = tf.Session(config=self.config)
        sess.run([self.init_global,
                  self.init_local],
                 feed_dict={})
        return sess

    def fit(self, data):
        """Train Model to fit data.

        Args:
            data ([dict]): have data and labels.
        """
        sess = self.init_session()

        subset_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="w_c")
        saver = tf.train.Saver(subset_variables)
        if self.restore:
            saver.restore(sess, self.checkpoint_fullpath_subset)

        subset_variables += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="w_enc") + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="w_dec")
        logger.debug(subset_variables)
        saver_fullmodel = tf.train.Saver(subset_variables)

        train_loss_summary_op = tf.summary.scalar("train_loss/euclid_distance", self.loss)
        test_loss_summary_op = tf.summary.scalar("test_loss/euclid_distance", self.loss)
        share_variables_summary_op = tf.summary.merge_all(key="share_variables")

        init_t = time.time()
        epoch = 0

        while epoch < self.n_epochs:
            # init
            train_loss = 0.
            test_loss = 0.

            ### Training steps ###

            batcher = self.minibatcher(data["train_x"])

            for batch, (x_previous, x_now, x_next) in enumerate(batcher):


                feed_dict = {
                    self.x_previous: x_previous,
                    self.x_now: x_now,
                    self.x_next: x_next,
                    self.learning_rate: self.lr
                }

                __, loss_value, loss_summary, share_summary = sess.run(
                    [self.train_op, self.loss, train_loss_summary_op, share_variables_summary_op], feed_dict=feed_dict)
                train_loss += loss_value
                self.train_summary_writer.add_summary(loss_summary, batch)

            train_loss /= (batch + 1.)

            ### Test steps ###

            batcher = self.minibatcher_withlabel(data["test_x"], data["test_label"])

            self.logdir = "{}projector/in_test/{:0=2}/".format(self.default_logdir, epoch)
            os.makedirs(self.logdir, exist_ok=True)

            medium_op = list(
                map(
                    lambda tname: tf.get_default_graph().get_tensor_by_name(tname),
                    self.layers_name)
                )
            # def zero vec
            prev_context_summary_np = np.zeros((self.n_embedding, self.output_dim))
            next_context_summary_np = np.zeros((self.n_embedding, self.output_dim))
            prev_target_summary_np = np.zeros((self.n_embedding, self.output_dim))
            next_target_summary_np = np.zeros((self.n_embedding, self.output_dim))
            shapetarget_summary_np = np.zeros((self.n_embedding, self.output_dim))
            prev_context_img = np.zeros((self.n_embedding, self.size, self.size))
            next_context_img = np.zeros((self.n_embedding, self.size, self.size))
            target_img = np.zeros((self.n_embedding, self.size, self.size))
            shapetarget_img = np.zeros((self.n_embedding, self.size, self.size))

            labels = []

            for batch, (x_previous, x_now, x_next, x_label) in enumerate(batcher):

                feed_dict = {
                    self.x_previous: x_previous,
                    self.x_now: x_now,
                    self.x_next: x_next,
                    self.learning_rate: self.lr
                }

                loss_value, loss_summary, embedding_summary = sess.run([
                    self.loss,
                    test_loss_summary_op,
                    medium_op
                    ], feed_dict=feed_dict)
                test_loss += loss_value
                self.test_summary_writer.add_summary(loss_summary, batch)

                ## project embedding ##

                # select last layer output
                summary_np = np.array(embedding_summary[-1])
                # select first layer output
                summary_np2 = np.array(embedding_summary[0])

                # cat several vectors (output of model)
                prev_context_summary_np[batch: (batch+1)] = summary_np[self.constant_idx]
                next_context_summary_np[batch: (batch+1)] = summary_np[self.batchsize + self.constant_idx]
                prev_target_summary_np[batch: (batch+1)] = summary_np[2*self.batchsize + self.constant_idx]
                next_target_summary_np[batch: (batch+1)] = summary_np[3*self.batchsize + self.constant_idx]
                shapetarget_summary_np[batch: (batch+1)] = summary_np2[self.constant_idx]

                # cat several images
                prev_context_img[batch: (batch+1)] = x_previous[self.constant_idx]
                next_context_img[batch: (batch+1)] = x_next[self.constant_idx]
                target_img[batch: (batch+1)] = x_now[self.constant_idx]
                shapetarget_img[batch: (batch+1)] = x_now[self.constant_idx]

                labels.append(x_label)

            cat_tensors = np.concatenate([
                prev_context_summary_np,
                next_context_summary_np,
                prev_target_summary_np,
                next_target_summary_np,
                shapetarget_summary_np
                ], axis=0)
            cat_images = np.concatenate([
                prev_context_img,
                next_context_img,
                target_img,
                target_img,
                shapetarget_img
                ], axis=0)
            cat_labels = labels + labels + labels + labels + labels

            variables = tf.Variable(cat_tensors, trainable=False, name="embedding")

            self.create_tensor_tsv(cat_tensors, path="{}{}".format(self.logdir, self.tensor_path_name))

            self.create_metadata_csv(cat_labels, path="{}{}".format(self.logdir, self.metadata_path_name))

            self.mk_spriteimage(cat_images, path="{}{}".format(self.logdir, self.sprite_image_path_name))

            config_projector = self.set_config_projector(variables)
            summary_writer = tf.summary.FileWriter(self.logdir, sess.graph)
            projector.visualize_embeddings(summary_writer, config_projector)

            # update the learning rate
            if epoch % 3 == 0 and epoch < 7:#warm up
                self.lr *= 1.25
            if epoch % 12 == 0 and epoch > 7:
                self.lr = self.lr * 0.75

            test_loss /= (batch + 1.)

            # save to wandb
            wandb.log({'epochs': epoch, 'loss': train_loss, 'test_loss': test_loss, 'learning_rate': self.lr})

            # save models
            saver_fullmodel.save(sess, self.checkpoint)

            epoch += 1

            logger.debug('[{:04d} | {:04.1f}] Train loss: {:04.8f}, Test loss: {:04.8f}'.format(epoch, time.time() - init_t, train_loss, test_loss))

        sess.close()

    def predict(self, sess, data, epoch):

        self.logdir = "{}projector/after_test/{:0=2}/".format(self.default_logdir, epoch)
        os.makedirs(self.logdir, exist_ok=True)

        medium_op = list(
            map(
                lambda tname: tf.get_default_graph().get_tensor_by_name(tname),
                self.layers_name)
            )

        batcher = self.minibatcher_withlabel(data["test_x"], data["test_label"])

        # def zero vec
        prev_context_summary_np = np.zeros((self.n_embedding, self.output_dim))
        next_context_summary_np = np.zeros((self.n_embedding, self.output_dim))
        prev_target_summary_np = np.zeros((self.n_embedding, self.output_dim))
        next_target_summary_np = np.zeros((self.n_embedding, self.output_dim))
        shapetarget_summary_np = np.zeros((self.n_embedding, self.output_dim))
        prev_context_img = np.zeros((self.n_embedding, self.size, self.size))
        next_context_img = np.zeros((self.n_embedding, self.size, self.size))
        target_img = np.zeros((self.n_embedding, self.size, self.size))
        shapetarget_img = np.zeros((self.n_embedding, self.size, self.size))

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
            prev_context_summary_np[i: (i+1)] = summary_np[self.constant_idx]
            next_context_summary_np[i: (i+1)] = summary_np[self.batchsize + self.constant_idx]
            prev_target_summary_np[i: (i+1)] = summary_np[2*self.batchsize + self.constant_idx]
            next_target_summary_np[i: (i+1)] = summary_np[3*self.batchsize + self.constant_idx]
            shapetarget_summary_np[i: (i+1)] = summary_np2[self.constant_idx]

            # cat several images
            prev_context_img[i: (i+1)] = x_previous[self.constant_idx]
            next_context_img[i: (i+1)] = x_next[self.constant_idx]
            target_img[i: (i+1)] = x_now[self.constant_idx]
            shapetarget_img[i: (i+1)] = x_now[self.constant_idx]

            labels.append(x_label)

        cat_tensors = np.concatenate([
            prev_context_summary_np,
            next_context_summary_np,
            prev_target_summary_np,
            next_target_summary_np,
            shapetarget_summary_np
            ], axis=0)
        cat_images = np.concatenate([
            prev_context_img,
            next_context_img,
            target_img,
            target_img,
            shapetarget_img
            ], axis=0)
        cat_labels = labels + labels + labels + labels + labels

        # tensorize
        variables = tf.Variable(cat_tensors, trainable=False, name="embedding")

        self.create_tensor_tsv(cat_tensors, path="{}{}".format(self.logdir, self.tensor_path_name))

        self.create_metadata_csv(cat_labels, path="{}{}".format(self.logdir, self.metadata_path_name))

        self.mk_spriteimage(cat_images, path="{}{}".format(self.logdir, self.sprite_image_path_name))

        config_projector = self.set_config_projector(variables)
        summary_writer = tf.summary.FileWriter(self.logdir, sess.graph)
        projector.visualize_embeddings(summary_writer, config_projector)

    def minibatcher(self, inputs):
        """Yield batchsize data.

        Args:
            inputs (list): list of data path. (*.npy)
            shuffle (bool, optional): shffule idx. Defaults to False.

        Yields:
            x_previous, x_now, x_next (ndarray):
        """
        size = self.size
        bs = self.batchsize

        for idx in range(0, len(inputs), bs):

            x = inputs[idx: idx + bs]

            if self.dataset_name == "morph":

                for m in range(0, x.shape[1]-2):
                    x_previous = np.reshape(x[:, 0+m], (bs, size, size))
                    x_now = np.reshape(x[:, 1+m], (bs, size, size))
                    x_next = np.reshape(x[:, 2+m], (bs, size, size))

                    yield x_previous, x_now, x_next

            elif self.dataset_name == "minimorph":

                x_previous = np.reshape(x[:, 0], (bs, size, size))
                x_now = np.reshape(x[:, 1], (bs, size, size))
                x_next = np.reshape(x[:, 2], (bs, size, size))

                yield x_previous, x_now, x_next

    def minibatcher_withlabel(self, inputs, labels):
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

    def create_tensor_tsv(self, cat_tensors, path):
        # make tensor.tsv
        df = pd.DataFrame(cat_tensors).astype("float64")
        df.to_csv(path, header=False, index=False, sep="\t")

    def create_metadata_csv(self, cat_labels, path):
        # make metadata.tsv (labels)
        with open(path, "w") as f:
            f.write("Index\tLabel\tPrevContext_NextContext_PrevTarget_NextTarget_ShapeTarget\n")

            for index, label in enumerate(cat_labels):
                id_label = self.label_dict[label]

                multi_label = index // self.n_embedding

                f.write("%d\t%d\t%d\n" % (index, id_label, multi_label))

    def mk_spriteimage(self, cat_images, path):
        # make sprite image (labels)
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
        embedding.sprite.single_image_dim.extend([self.size, self.size])
        return config_projector
