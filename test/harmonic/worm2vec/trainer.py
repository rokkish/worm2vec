"""Trainer class mainly train model
"""
import os
import sys
import time
import tensorflow as tf
import numpy as np
import pandas as pd
from post_slack import post
import get_logger
logger = get_logger.get_logger(name="trainer")


class Trainer():
    def __init__(self,
                 params,
                 loss,
                 valid_loss,
                 optim,
                 train_op,
                 placeholders):
        self.lr = params.optimizer.learning_rate
        self.n_epochs = params.nn.n_epochs
        self.batch_size = params.nn.batch_size
        self.checkpoint_path = params.path.checkpoint_model

        self.loss = loss
        self.optim = optim
        self.train_op = train_op

        self.positive = placeholders["positive"]
        self.negative = placeholders["negative"]
        self.learning_rate = placeholders["learning_rate"]
        self.train_phase = placeholders["train_phase"]

        # Configure tensorflow session
        self.init_global = tf.global_variables_initializer()
        self.init_local = tf.local_variables_initializer()
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = params.gpu.allow_growth
        self.config.gpu_options.visible_device_list = params.gpu.id
        self.config.log_device_placement = params.gpu.log_device_placement

        # Restart train
        self.restart_train = params.train.restart_train
        self.checkpoint_fullpath = params.path.checkpoint_fullpath

        # num of pos, neg
        self.n_positive = params.nn.n_positive
        self.n_negative = params.nn.n_negative

        # validation loss not trainable
        self.cossim = valid_loss

        # summary_writer
        self.train_summary_writer = tf.summary.FileWriter("./tensorboard/train")
        self.valid_summary_writer = tf.summary.FileWriter("./tensorboard/valid")
        self.test_summary_writer = tf.summary.FileWriter("./tensorboard/test")

        # save test_score
        self.loss_tocsv = {"train_loss": [], "valid_pp": [], "valid_pn": [], "test_pp": [], "test_pn": [], "train_pp": [], "train_pn": []}
        self.csv_path = params.path.test_score

    def fit(self, data):
        saver = tf.train.Saver()
        sess = tf.Session(config=self.config)
        sess.run([self.init_global,
                  self.init_local],
                 feed_dict={self.train_phase: True})

        if self.restart_train:
            saver.restore(sess, self.checkpoint_fullpath)

        start = time.time()
        epoch = 0
        logger.debug('Starting training loop...')

        anchorloss_sum = tf.summary.scalar("anchor_loss/pos_neg", self.loss)
        train_cossim_sum = self.cossim_summary("train")
        valid_cossim_sum = self.cossim_summary("valid")
        test_cossim_sum = self.cossim_summary("test")

        while epoch < self.n_epochs:
            # Training steps
            batcher = \
                self.minibatcher(
                    data['train_x'],
                    self.batch_size,
                    shuffle=True)
            anchor_loss = 0.
            train_cossim_dict = {}
            train_cossim_dict["pp"], train_cossim_dict["pn"] = 0., 0.
            for i, (Pos, Neg) in enumerate(batcher):
                feed_dict = {self.positive: Pos,
                            self.negative: Neg,
                            self.learning_rate: self.lr,
                            self.train_phase: True}
                __, loss, anchor_summary, cossim, cossim_summary = sess.run([
                                    self.train_op,
                                    self.loss,
                                    anchorloss_sum,
                                    self.cossim,
                                    train_cossim_sum],
                                    feed_dict=feed_dict)
                anchor_loss += loss
                self.train_summary_writer.add_summary(anchor_summary, i)
                train_cossim_dict["pp"] += cossim["pp"]
                train_cossim_dict["pn"] += cossim["pn"]
                self.train_summary_writer.add_summary(cossim_summary, i)

            anchor_loss /= (i+1.)
            train_cossim_dict["pp"] /= (i+1.)
            train_cossim_dict["pn"] /= (i+1.)

            # Validation steps
            batcher = \
                self.minibatcher(
                    data['valid_x'],
                    self.batch_size)
            valid_cossim_dict = {}
            valid_cossim_dict["pp"], valid_cossim_dict["pn"] = 0., 0.
            for i, (Pos, Neg) in enumerate(batcher):
                feed_dict = {self.positive: Pos,
                            self.negative: Neg,
                            self.learning_rate: self.lr,
                            self.train_phase: False}
                cossim, cossim_summary = sess.run([self.cossim, valid_cossim_sum], feed_dict=feed_dict)
                valid_cossim_dict["pp"] += cossim["pp"]
                valid_cossim_dict["pn"] += cossim["pn"]

                sys.stdout.write('Validating\r')
                sys.stdout.flush()

                self.valid_summary_writer.add_summary(cossim_summary, i)

            valid_cossim_dict["pp"] /= (i+1.)
            valid_cossim_dict["pn"] /= (i+1.)

            # Test steps
            batcher = \
                self.minibatcher(
                    data['test_x'],
                    self.batch_size)
            test_cossim_dict = {}
            test_cossim_dict["pp"], test_cossim_dict["pn"] = 0., 0.
            for i, (Pos, Neg) in enumerate(batcher):
                feed_dict = {self.positive: Pos,
                            self.negative: Neg,
                            self.learning_rate: self.lr,
                            self.train_phase: False}
                cossim, cossim_summary = sess.run([self.cossim, test_cossim_sum], feed_dict=feed_dict)
                test_cossim_dict["pp"] += cossim["pp"]
                test_cossim_dict["pn"] += cossim["pn"]

                sys.stdout.write('Testing\r')
                sys.stdout.flush()

                self.test_summary_writer.add_summary(cossim_summary, i)

            test_cossim_dict["pp"] /= (i+1.)
            test_cossim_dict["pn"] /= (i+1.)

            # save loss
            self.loss_tocsv["train_loss"].append(anchor_loss)
            self.loss_tocsv["train_pp"].append(train_cossim_dict["pp"])
            self.loss_tocsv["train_pn"].append(train_cossim_dict["pn"])
            self.loss_tocsv["valid_pp"].append(valid_cossim_dict["pp"])
            self.loss_tocsv["valid_pn"].append(valid_cossim_dict["pn"])
            self.loss_tocsv["test_pp"].append(test_cossim_dict["pp"])
            self.loss_tocsv["test_pn"].append(test_cossim_dict["pn"])

            # Save model
            if epoch % 10 == 0 or epoch == self.n_epochs - 1:
                saver.save(sess, self.checkpoint_path)
                logger.debug('Model saved: {}'.format(self.checkpoint_path))

            # Updates to the training scheme
            if epoch % 4 == 0:
                self.lr = self.lr * np.power(0.1, epoch / 50)
            epoch += 1

            log_tmp = '[{:04d} | {:04.1f}] Train anchor loss: {:04.8f}, Learning rate: {:.2e}'.format(epoch, time.time()-start, anchor_loss, self.lr)
            logger.info(log_tmp)

        # Save loss
        df = pd.DataFrame(self.loss_tocsv)
        df.to_csv(self.csv_path, mode="a", header=not os.path.exists(self.csv_path))

        sess.close()

    def minibatcher(self, inputs, batchsize, shuffle=False):
        """

        Args:
            inputs (ndarray): (N, 41, 1, 64, 64)
            batchsize (int): [description]
            shuffle (bool, optional): [description]. Defaults to False.

        Yields:
            positive, negative (ndarray): for proxy-anchor loss.
        """
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs), 1):
            if shuffle:
                excerpt = indices[start_idx]
            else:
                excerpt = start_idx

            yield inputs[excerpt, :self.n_positive, 0], inputs[excerpt, -self.n_negative:, 0]

    def cossim_summary(self, mode):
        """
        Args:
            mode: [train, valid, test]
        Return:
            tf.summary
        """
        if mode not in ["train", "valid", "test"]:
            raise ValueError("mode must be [train, valid, test]")

        cossim_sum1 = tf.summary.scalar(mode + "_loss/pp", self.cossim["pp"])
        cossim_sum2 = tf.summary.scalar(mode + "_loss/pn", self.cossim["pn"])
        cossim_sum3 = tf.summary.scalar(mode + "_loss/nn", self.cossim["nn"])
        return tf.summary.merge([cossim_sum1, cossim_sum2, cossim_sum3])
