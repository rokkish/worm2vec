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

        # save loss
        self.loss_tocsv = {
            "train_loss": [],
            "test_loss": []
            }

        # Configure tensorflow session
        self.init_global = tf.global_variables_initializer()
        self.init_local = tf.local_variables_initializer()
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = params.device.gpu.allow_growth
        self.config.gpu_options.visible_device_list = params.device.gpu.id
        self.config.log_device_placement = params.device.gpu.log_device_placement

    def init_session(self):
        """Initial tensorflow Session

        Returns:
            saver: load self.config
            sess: load inital config
        """
        saver = tf.train.Saver()
        sess = tf.Session(config=self.config)
        sess.run([self.init_global,
                  self.init_local],
                 feed_dict={})
        return saver, sess

    def fit(self, data):
        """Train Model to fit data.

        Args:
            data ([dict]): have data and labels.
        """
        saver, sess = self.init_session()

        if self.restore:
            subset_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="w_c")
            logger.debug(subset_variables)
            saver_ = tf.train.Saver(subset_variables)
            saver_.restore(sess, self.checkpoint_fullpath_subset)

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
                if batch % 40 == 0:
                    self.train_summary_writer.add_summary(share_summary, batch)

            ### Test steps ###

            batcher = self.minibatcher(data["test_x"])

            for batch, (x_previous, x_now, x_next) in enumerate(batcher):

                feed_dict = {
                    self.x_previous: x_previous,
                    self.x_now: x_now,
                    self.x_next: x_next,
                    self.learning_rate: self.lr
                }

                loss_value, loss_summary = sess.run([self.loss, test_loss_summary_op], feed_dict=feed_dict)
                test_loss += loss_value
                self.test_summary_writer.add_summary(loss_summary, batch)

            # update the learning rate
            if epoch % 4 == 0:
                self.lr = self.lr * 0.25

            # save loss to csv
            train_loss /= (batch + 1.)
            test_loss /= (batch + 1.)
            self.loss_tocsv["train_loss"].append(train_loss)
            self.loss_tocsv["test_loss"].append(test_loss)

            # save to wandb
            wandb.log({'epochs': epoch, 'loss': train_loss, 'test_loss': test_loss, 'learning_rate': self.lr})

            # save models
            saver.save(sess, self.checkpoint)

            epoch += 1

            logger.debug('[{:04d} | {:04.1f}] Train loss: {:04.8f}, Test loss: {:04.8f}'.format(epoch, time.time() - init_t, train_loss, test_loss))

        # save loss to df
        pd.DataFrame(self.loss_tocsv).to_csv(self.csv_path, mode="a", header=not os.path.exists(self.csv_path))

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

            x_previous = np.reshape(x[:, 0], (bs, size, size))
            x_now = np.reshape(x[:, 1], (bs, size, size))
            x_next = np.reshape(x[:, 2], (bs, size, size))

            yield x_previous, x_now, x_next
