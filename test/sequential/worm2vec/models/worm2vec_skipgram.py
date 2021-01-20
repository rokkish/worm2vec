"""predict mean of 2d-shapes like Skip-gram
"""
import logging
import tensorflow as tf
import sys
sys.path.append("..")
import get_logger
logger = get_logger.get_logger(name="skipgram")


class skipgram(object):
    def __init__(self, x_previous, x_next, x_now, input_dim):
        self.x_previous = x_previous
        self.x_next = x_next
        self.x_now = x_now
        self.input_dim = input_dim
        # set out_dim is input_dim
        self.latent_dim = self.input_dim

        w_initializer = tf.initializers.he_normal()

        self.W_enc_target = tf.get_variable(
            'w_enc_target',
            shape=[self.input_dim, self.latent_dim],
            dtype=tf.float32,
            initializer=w_initializer)
        self.W_enc_prev = tf.get_variable(
            'w_enc_prev',
            shape=[self.latent_dim, self.latent_dim],
            dtype=tf.float32,
            initializer=w_initializer)
        self.W_enc_next = tf.get_variable(
            'w_enc_next',
            shape=[self.latent_dim, self.latent_dim],
            dtype=tf.float32,
            initializer=w_initializer)

    def nn(self):
        """Simple CNN model for prediction"""

        with tf.name_scope("concat_inputs"):
            x_target = self.x_now
            x_context = tf.concat([self.x_previous, self.x_next], axis=0)

        with tf.name_scope("context_encoder"):
            z_target = tf.matmul(x_target, self.W_enc_target)
        with tf.name_scope("prev_encoder"):
            prev_target = tf.matmul(z_target, self.W_enc_prev)
        with tf.name_scope("next_encoder"):
            next_target = tf.matmul(z_target, self.W_enc_next)

        with tf.name_scope("concat_outputs"):
            return tf.concat([x_context, prev_target, next_target], axis=0)


def cosine_similarity(x, y):
    with tf.name_scope('cosine_similarity'):
        return tf.nn.l2_normalize(x, axis=-1) * tf.nn.l2_normalize(y, axis=-1)


def skipgram_loss(context, target):
    """euclid distance loss between context and target"""
    logger.debug("context shape: {}".format(context.shape))
    with tf.name_scope('prev_context'):
        prev_context = context[:int(context.shape[0])//2]
    with tf.name_scope('next_context'):
        next_context = context[int(context.shape[0])//2:]
    with tf.name_scope('prev_target'):
        prev_target = target[:int(target.shape[0])//2]
    with tf.name_scope('next_target'):
        next_target = target[int(target.shape[0])//2:]

    with tf.name_scope('distance_loss'):
        distance = cosine_similarity(prev_context, prev_target)
        distance += cosine_similarity(next_context, next_target)
        loss = 1. - tf.reduce_mean(tf.reduce_sum(distance/2., axis=-1))

    return loss
