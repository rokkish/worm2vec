"""loss for metric learning
"""
import sys
sys.path.append('../')

import tensorflow as tf
import numpy as np
import get_logger
logger = get_logger.get_logger(name='losses')


def cosine_similarity(a, b):
    return a * b / (tf.nn.l2_normalize(a, axis=1) * tf.nn.l2_normalize(b, axis=1))


def cosine_similarity_pos_neg(embeddings):
    pos = embeddings["positive"]
    neg = embeddings["negative"]
    arr = tf.concat([pos, neg], axis=0)

    norm_arr = tf.nn.l2_normalize(arr, axis=1)
    expand1 = tf.expand_dims(norm_arr, axis=0)
    expand2 = tf.expand_dims(norm_arr, axis=1)
    cos_matrix = tf.reduce_sum(expand1 * expand2, axis=-1)

    cos_matrix_pp = cos_matrix[: pos.shape[0],              : pos.shape[0]]
    cos_matrix_pn = cos_matrix[: pos.shape[0],  pos.shape[0]:             ]
    cos_matrix_nn = cos_matrix[pos.shape[0]: ,  pos.shape[0]:             ]

    loss = tf.reduce_mean(tf.matrix_band_part(cos_matrix, -1, 0))
    loss_pp = tf.reduce_mean(tf.matrix_band_part(cos_matrix_pp, -1, 0))
    loss_pn = tf.reduce_mean(tf.matrix_band_part(cos_matrix_pn, -1, 0))
    loss_nn = tf.reduce_mean(tf.matrix_band_part(cos_matrix_nn, -1, 0))

    return {"all": loss, "pp": loss_pp, "pn": loss_pn, "nn": loss_nn}


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

    pos_sim_mat = tf.matmul(pos_embeddings_l2, proxy_l2, transpose_b=True)
    neg_sim_mat = tf.matmul(neg_embeddings_l2, proxy_l2, transpose_b=True)

    pos_mat = tf.exp(-alpha * (pos_sim_mat - delta))
    neg_mat = tf.exp(alpha * (neg_sim_mat + delta))

    # n_unique = batch_size // n_instance
    pos_term = 1.0 / n_unique * tf.reduce_sum(tf.log(1.0 + tf.reduce_sum(pos_mat, axis=0)))
    neg_term = 1.0 / n_classes * tf.reduce_sum(tf.log(1.0 + tf.reduce_sum(neg_mat, axis=0)))

    loss = pos_term + neg_term

    return loss
