"""loss for metric learning
"""
import sys
sys.path.append('../')

import tensorflow as tf
import get_logger
logger = get_logger.get_logger(name='losses')


def compute_euclidian_distance(x, y):
    return tf.reduce_sum(tf.square(x - y), 1)


def triplet_loss(preds, margin):
    anchor = preds["x"]
    positive = preds["positive"]
    negative = preds["negative"]

    if len(anchor.shape) != 2:
        raise ValueError("tensor shape should be [batch, dim]")

    dist_pos = compute_euclidian_distance(anchor, positive)
    dist_neg = compute_euclidian_distance(anchor, negative)

    loss = tf.reduce_mean(tf.maximum(0., margin + dist_pos - dist_neg))
    return loss


def proxy_anchor_loss(embeddings, n_classes, n_unique, input_dim, alpha, delta):
    '''
    ref: https://arxiv.org/abs/2003.13911
    official pytorch codes: https://github.com/tjddus9597/Proxy-Anchor-CVPR2020
    '''
    # define proxy weights
    proxy = tf.get_variable(name='proxy', shape=[n_classes, input_dim],
                            initializer=tf.random_normal_initializer(),
                            dtype=tf.float32,
                            trainable=True)
    pos_embeddings_l2 = tf.nn.l2_normalize(embeddings["positive"], axis=1)
    neg_embeddings_l2 = tf.nn.l2_normalize(embeddings["negative"], axis=1)
    proxy_l2 = tf.nn.l2_normalize(proxy, axis=1)

    logger.debug("proxy shape: {}".format(proxy_l2.shape))
    logger.debug("  pos shape: {}".format(pos_embeddings_l2.shape))

    pos_sim_mat = tf.matmul(pos_embeddings_l2, proxy_l2, transpose_b=True)
    neg_sim_mat = tf.matmul(neg_embeddings_l2, proxy_l2, transpose_b=True)

    pos_mat = tf.exp(-alpha * (pos_sim_mat - delta))
    neg_mat = tf.exp(alpha * (neg_sim_mat + delta))

    logger.debug("sim_mat shape: {}".format(pos_sim_mat.shape))

    # n_unique = batch_size // n_instance
    pos_term = 1.0 / n_unique * tf.reduce_sum(tf.log(1.0 + tf.reduce_sum(pos_mat, axis=0)))
    neg_term = 1.0 / n_classes * tf.reduce_sum(tf.log(1.0 + tf.reduce_sum(neg_mat, axis=0)))

    loss = pos_term + neg_term

    return loss
