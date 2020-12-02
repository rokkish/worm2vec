"""Trainer class mainly train model
"""
import os
import time
import tensorflow as tf
import numpy as np
import pandas as pd
import get_logger
logger = get_logger.get_logger(name="trainer")


class Trainer(object):
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
        self.checkpoint = params.dir.checkpoint
        self.csv_path = params.dir.losscsv
        self.dim = params.training.dim

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
        saver = tf.train.Saver()
        sess = tf.Session(config=self.config)
        sess.run([self.init_global,
                  self.init_local],
                 feed_dict={})
        return saver, sess

    def fit(self, data):
        saver, sess = self.init_session()

        train_loss_summary_op = tf.summary.scalar("train_loss/euclid_distance", self.loss)
        test_loss_summary_op = tf.summary.scalar("test_loss/euclid_distance", self.loss)

        init_t = time.time()
        epoch = 0

        while epoch < self.n_epochs:
            # init
            train_loss = 0.
            test_loss = 0.

            ### Training steps ###

            batcher = self.minibatcher(data["train_x"], shuffle=True)

            for batch, (x_previous, x_now, x_next) in enumerate(batcher):

                if batch == 0:
                    logger.debug(x_previous.shape)
                    logger.debug(x_now.shape)
                    logger.debug(x_next.shape)

                feed_dict = {
                    self.x_previous: x_previous,
                    self.x_now: x_now,
                    self.x_next: x_next,
                    self.learning_rate: self.lr
                }

                __, loss_value, loss_summary = sess.run(
                    [self.train_op, self.loss, train_loss_summary_op], feed_dict=feed_dict)
                train_loss += loss_value
                self.train_summary_writer.add_summary(loss_summary, batch)

            ### Test steps ###

            batcher = self.minibatcher(data["test_x"], shuffle=True)

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

            # save loss to csv
            train_loss /= (batch + 1.)
            test_loss /= (batch + 1.)
            self.loss_tocsv["train_loss"].append(train_loss)
            self.loss_tocsv["test_loss"].append(test_loss)

            # save models
            saver.save(sess, self.checkpoint)

            epoch += 1

            logger.info('[{:04d} | {:04.1f}] Train loss: {:04.8f}'.format(epoch, time.time() - init_t, train_loss))

        # save loss to df
        pd.DataFrame(self.loss_tocsv).to_csv(self.csv_path, mode="a", header=not os.path.exists(self.csv_path))

    def minibatcher(self, inputs, shuffle=False):
        """

        Args:
            inputs (list): list of data path. (*.npy)
            shuffle (bool, optional): shffule idx. Defaults to False.

        Yields:
            x_previous, x_now, x_next (ndarray):
        """
        dim = self.dim

        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)

        for start_idx in range(0, len(inputs), 1):
            if shuffle:
                excerpt = indices[start_idx]
            else:
                excerpt = start_idx

            x = np.load(inputs[excerpt])

            yield np.reshape(x[0, 0], (1, dim, dim)), np.reshape(x[0, 5], (1, dim, dim)), np.reshape(x[0, -1], (1, dim, dim))
