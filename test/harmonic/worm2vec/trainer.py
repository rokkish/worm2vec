"""Trainer class mainly train model
"""
import sys
import time
import tensorflow as tf
import numpy as np
import get_logger
logger = get_logger.get_logger(name="trainer")


class Trainer():
    def __init__(self,
                 params,
                 loss,
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

        self.x = placeholders["x"]
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

        while epoch < self.n_epochs:
            # Training steps
            batcher = \
                self.minibatcher(
                    data['train_x'],
                    self.batch_size,
                    shuffle=True)
            train_loss = 0.
            for i, (X, Pos, Neg) in enumerate(batcher):
                feed_dict = {self.x: X,
                             self.positive: Pos,
                             self.negative: Neg,
                             self.learning_rate: self.lr,
                             self.train_phase: True}
                __, loss_ = sess.run([self.train_op,
                                      self.loss],
                                     feed_dict=feed_dict)
                train_loss += loss_
                sys.stdout.write('{}/{}\r'.format(i, data['train_x'].shape[0]/self.batch_size))
                sys.stdout.flush()
            train_loss /= (i+1.)

            # Save model
            if epoch % 10 == 0:
                saver.save(sess, self.checkpoint_path)
                print('Model saved')

            # Updates to the training scheme
            self.lr = self.lr * np.power(0.1, epoch / 50)
            epoch += 1

            logger.info('[{:04d} | {:04.1f}] Loss: {:04.4f}, Validation Loss.: {:04.4f}, Learning rate: {:.2e}'.format(epoch, time.time()-start, train_loss, valid_loss, self.lr))


        sess.close()

    @staticmethod
    def minibatcher(inputs, batchsize, shuffle=False):
        """

        Args:
            inputs (ndarray): (N, 41, 1, 64, 64)
            batchsize (int): [description]
            shuffle (bool, optional): [description]. Defaults to False.

        Yields:
            anchor, positive, negative (ndarray): for triplet loss.
        """
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            rotation = 17 #1~35
            negative = rotation + 1 #1~5
            yield inputs[excerpt, 0, 0], inputs[excerpt, rotation, 0], inputs[excerpt, negative, 0]
