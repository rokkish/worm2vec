"""Trainer class mainly train model
"""
import sys
import time
import tensorflow as tf
import numpy as np
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
        self.valid_loss = valid_loss

        # summary_writer
        self.train_summary_writer = tf.summary.FileWriter("./tensorboard/train")
        self.valid_summary_writer = tf.summary.FileWriter("./tensorboard/valid")
        self.test_summary_writer = tf.summary.FileWriter("./tensorboard/test")

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

        anchor_loss_sum = tf.summary.scalar("anchor_loss/pos_neg", self.loss)
        loss_sum0 = tf.summary.scalar("train_loss/all", self.valid_loss["all"])
        loss_sum1 = tf.summary.scalar("train_loss/pp", self.valid_loss["pp"])
        loss_sum2 = tf.summary.scalar("train_loss/pn", self.valid_loss["pn"])
        loss_sum3 = tf.summary.scalar("train_loss/nn", self.valid_loss["nn"])
        train_loss_sum = tf.summary.merge([loss_sum0, loss_sum1, loss_sum2, loss_sum3])
        loss_sum0 = tf.summary.scalar("valid_loss/all", self.valid_loss["all"])
        loss_sum1 = tf.summary.scalar("valid_loss/pp", self.valid_loss["pp"])
        loss_sum2 = tf.summary.scalar("valid_loss/pn", self.valid_loss["pn"])
        loss_sum3 = tf.summary.scalar("valid_loss/nn", self.valid_loss["nn"])
        valid_loss_sum = tf.summary.merge([loss_sum0, loss_sum1, loss_sum2, loss_sum3])
        loss_sum0 = tf.summary.scalar("test_loss/all", self.valid_loss["all"])
        loss_sum1 = tf.summary.scalar("test_loss/pp", self.valid_loss["pp"])
        loss_sum2 = tf.summary.scalar("test_loss/pn", self.valid_loss["pn"])
        loss_sum3 = tf.summary.scalar("test_loss/nn", self.valid_loss["nn"])
        test_loss_sum = tf.summary.merge([loss_sum0, loss_sum1, loss_sum2, loss_sum3])

        while epoch < self.n_epochs:
            # Training steps
            batcher = \
                self.minibatcher(
                    data['train_x'],
                    self.batch_size,
                    shuffle=True)
            anchor_loss = 0.
            for i, (Pos, Neg) in enumerate(batcher):
                feed_dict = {self.positive: Pos,
                            self.negative: Neg,
                            self.learning_rate: self.lr,
                            self.train_phase: True}
                __, loss, loss_ = sess.run([
                                    self.train_op,
                                    self.loss,
                                    anchor_loss_sum],
                                    feed_dict=feed_dict)
                anchor_loss += loss
                self.train_summary_writer.add_summary(loss_, i)
            anchor_loss /= (i+1.)

            # Validation steps
            batcher = \
                self.minibatcher(
                    data['valid_x'],
                    self.batch_size)
            for i, (Pos, Neg) in enumerate(batcher):
                feed_dict = {self.positive: Pos,
                            self.negative: Neg,
                            self.learning_rate: self.lr,
                            self.train_phase: False}
                loss_, result = sess.run([self.valid_loss, valid_loss_sum], feed_dict=feed_dict)

                sys.stdout.write('Validating\r')
                sys.stdout.flush()

                self.valid_summary_writer.add_summary(result, i)

            # Test steps
            batcher = \
                self.minibatcher(
                    data['test_x'],
                    self.batch_size)
            for i, (Pos, Neg) in enumerate(batcher):
                feed_dict = {self.positive: Pos,
                            self.negative: Neg,
                            self.learning_rate: self.lr,
                            self.train_phase: False}
                loss_, result = sess.run([self.valid_loss, test_loss_sum], feed_dict=feed_dict)

                sys.stdout.write('Testing\r')
                sys.stdout.flush()

                self.test_summary_writer.add_summary(result, i)

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
            post(log_tmp)

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
