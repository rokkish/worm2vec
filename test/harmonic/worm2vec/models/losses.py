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


def mean_nondiag(matrix, matrix_shape):
    with tf.name_scope('mean_nondiag_part'):
        sum_lower = tf.reduce_sum(tf.matrix_band_part(matrix, -1, 0))
        sum_diag = tf.reduce_sum(tf.matrix_diag_part(matrix))
        return (sum_lower - sum_diag) / (matrix_shape**2 - matrix_shape) * 2


def cosine_similarity_pos_neg(embeddings):
    pos = embeddings["positive"]
    neg = embeddings["negative"]
    arr = tf.concat([pos, neg], axis=0)

    with tf.name_scope('cosine_matrix'):
        norm_arr = tf.nn.l2_normalize(arr, axis=1)
        expand1 = tf.expand_dims(norm_arr, axis=0)
        expand2 = tf.expand_dims(norm_arr, axis=1)
        cos_matrix = tf.reduce_sum(expand1 * expand2, axis=-1)

    with tf.name_scope('cosine_matrix_pairs'):
        cos_matrix_pp = cos_matrix[: pos.shape[0],              : pos.shape[0]]
        cos_matrix_pn = cos_matrix[: pos.shape[0],  pos.shape[0]:             ]
        cos_matrix_nn = cos_matrix[pos.shape[0]: ,  pos.shape[0]:             ]

    with tf.name_scope('cosine_similarity'):
        loss = tf.reduce_mean(cos_matrix)
        loss_pp = mean_nondiag(cos_matrix_pp, int(pos.shape[0]))
        loss_pn = tf.reduce_mean(cos_matrix_pn)
        loss_nn = mean_nondiag(cos_matrix_nn, int(neg.shape[0]))
    #ret = {"pp": tf.matrix_band_part(cos_matrix_pp, -1, 0),
    #       "nn":tf.matrix_band_part(cos_matrix_nn, -1, 0),
    #       "nn_diag": tf.matrix_diag_part(cos_matrix_nn)}

    return {"all": loss, "pp": loss_pp, "pn": loss_pn, "nn": loss_nn}


def euclidian_distance_pos_neg(embeddings):
    pos = embeddings["positive"]
    neg = embeddings["negative"]
    arr = tf.concat([pos, neg], axis=0)

    with tf.name_scope('euclid_matrix'):
        expand1 = tf.expand_dims(arr, axis=0)
        expand2 = tf.expand_dims(arr, axis=1)
        euclid_matrix = tf.reduce_sum(tf.square(expand1 - expand2), axis=-1)

    with tf.name_scope('euclid_matrix_pairs'):
        euclid_matrix_pp = euclid_matrix[: pos.shape[0], : pos.shape[0]]
        euclid_matrix_pn = euclid_matrix[: pos.shape[0],  pos.shape[0]:]
        euclid_matrix_nn = euclid_matrix[pos.shape[0]: ,  pos.shape[0]:]

    with tf.name_scope('euclid_distance'):
        loss = tf.reduce_mean(euclid_matrix)
        loss_pp = mean_nondiag(euclid_matrix_pp, int(pos.shape[0]))
        loss_pn = tf.reduce_mean(euclid_matrix_pn)
        loss_nn = mean_nondiag(euclid_matrix_nn, int(neg.shape[0]))
    #ret = {"pp": tf.matrix_band_part(euclid_matrix_pp, -1, 0),
    #       "nn":tf.matrix_band_part(euclid_matrix_nn, -1, 0),
    #       "nn_diag": tf.matrix_diag_part(euclid_matrix_nn)}

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


def proxy_anchor_loss(embeddings, class_id, n_classes, n_unique, input_dim, alpha, delta):
    '''
    ref: https://arxiv.org/abs/2003.13911
    official pytorch codes: https://github.com/tjddus9597/Proxy-Anchor-CVPR2020
    '''
    # define proxy weights
    proxy = tf.get_variable(name='proxy', shape=[n_classes, input_dim],
                            initializer=tf.random_normal_initializer(),
                            dtype=tf.float32,
                            trainable=True)
    with tf.name_scope('select_proxy'):
        pos_proxy = tf.reshape(proxy[class_id], [1, input_dim])
    with tf.name_scope('l2_norm_pos'):
        pos_embeddings_l2 = tf.nn.l2_normalize(embeddings["positive"], axis=1)
    with tf.name_scope('l2_norm_neg'):
        neg_embeddings_l2 = tf.nn.l2_normalize(embeddings["negative"], axis=1)
    with tf.name_scope('l2_norm_proxy'):
        pos_proxy_l2 = tf.nn.l2_normalize(pos_proxy, axis=1)
        #all_proxy_l2 = tf.nn.l2_normalize(proxy, axis=1)

    with tf.name_scope('similarity_proxy_pos'):
        pos_sim_mat = tf.matmul(pos_embeddings_l2, pos_proxy_l2, transpose_b=True)
    with tf.name_scope('similarity_proxy_neg'):
        neg_sim_mat = tf.matmul(neg_embeddings_l2, pos_proxy_l2, transpose_b=True)
    with tf.name_scope('similarity_pos_neg'):
        pos_neg_sim_mat = tf.matmul(pos_embeddings_l2, neg_embeddings_l2, transpose_b=True)
    with tf.name_scope('similarity_pos_pos'):
        pos_pos_sim_mat = tf.matmul(pos_embeddings_l2, pos_embeddings_l2, transpose_b=True)

    with tf.name_scope('exp_of_similarity'):
        pos_mat = tf.exp(-alpha * (pos_sim_mat - delta))
        neg_mat = tf.exp(alpha * (neg_sim_mat + delta))
        pos_neg_mat = tf.exp(alpha * (pos_neg_sim_mat + delta))
        pos_pos_mat = tf.exp(-alpha * (pos_pos_sim_mat - delta))

    with tf.name_scope('softplus_logsumexp'):
        pos_term = tf.reduce_sum(tf.log(1.0 + tf.reduce_sum(pos_mat, axis=0)))
        neg_term = tf.reduce_sum(tf.log(1.0 + tf.reduce_sum(neg_mat, axis=0)))
        pos_neg_term = tf.reduce_mean(tf.log(1.0 + tf.reduce_mean(pos_neg_mat, axis=0)))
        pos_pos_term = tf.reduce_mean(tf.log(1.0 + tf.reduce_mean(pos_pos_mat, axis=0)))

    with tf.name_scope('proxy_anchor_loss'):
        loss = pos_term + neg_term + pos_neg_term + pos_pos_term

    with tf.name_scope('add_l2_loss'):
        var = tf.trainable_variables()
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in var if "get_weights" in v.name]) * 0.001
        loss += l2_loss
    return loss


def test_cosine_similarity_pos_neg():

    def set_placeholders(n_positive, n_negative, dim):
        positive = tf.placeholder(tf.float32,
                                [n_positive, dim],
                                name='positive')
        negative = tf.placeholder(tf.float32,
                                [n_negative, dim],
                                name='negative')
        return positive, negative

    def set_values(n_positive, n_negative, dim):
        if False:
            positive = np.zeros((n_positive, dim))
            negative = np.zeros((n_negative, dim))
            positive += 1
            negative += 1
        else:
            positive = np.random.normal(1.0, 0.001, (n_positive, dim))
            negative = np.random.normal(1.0, 0.1, (n_negative, dim))
        return positive, negative

    sess = tf.InteractiveSession()
    pos_shape, neg_shape, dim = 36, 50, 1000
    pos, neg = set_placeholders(pos_shape, neg_shape, dim)
    embeddings = {"positive": pos, "negative": neg}
    loss = euclidian_distance_pos_neg(embeddings)
    loss_log = {"pp": 0., "pn": 0., "nn": 0.}

    n_test = 100
    for i in range(n_test):
        Pos, Neg = set_values(pos_shape, neg_shape, dim)
        ret, loss_ = sess.run(loss, feed_dict={pos: Pos, neg: Neg})
        loss_log["pp"] += loss_["pp"] / n_test
        loss_log["pn"] += loss_["pn"] / n_test
        loss_log["nn"] += loss_["nn"] / n_test
    print("mean", loss_log)

    import pandas as pd
    df1 = pd.DataFrame(ret["pp"]).astype("float64")
    df2 = pd.DataFrame(ret["nn"]).astype("float64")
    df3 = pd.DataFrame(ret["nn_diag"]).astype("float64")
    df1.to_csv("./test_band_apart_pp.csv")
    df2.to_csv("./test_band_apart_nn.csv")
    df3.to_csv("./test_band_apart_nndiag.csv")
