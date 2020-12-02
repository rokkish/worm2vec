"""predict mean of 2d-shapes
"""
import tensorflow as tf


def nn(x_previous, x_next, x_now, input_dim, output_dim):
    """Simple CNN model for prediction"""

    context = x_previous.shape[0] + x_next.shape[0]
    N = context + x_now.shape[0]

    dim1 = input_dim//2
    dim2 = input_dim//4

    initializer = tf.constant_initializer(1e-2)
    W1 = tf.get_variable('w1', shape=[N, 1, input_dim, dim1], dtype=tf.float32, initializer=initializer)
    W2 = tf.get_variable('w2', shape=[N, 1, dim1, dim2], dtype=tf.float32, initializer=initializer)
    Wco = tf.get_variable('wco', shape=[dim2 ** 2, dim2 ** 2 //2], dtype=tf.float32, initializer=initializer)
    bias = tf.get_variable('b1', shape=[dim2 ** 2], initializer=initializer)

    with tf.name_scope("concat_inputs"):
        x = tf.concat([x_previous, x_next, x_now], axis=0)
        x = tf.reshape(x, shape=[N, 1, input_dim, input_dim])

    with tf.name_scope("share_encoder"):
        cv1 = tf.nn.conv2d(x, W1, strides=(1, 2, 2, 1), padding="SAME", name="conv1")
        cv1 = tf.nn.conv2d(cv1, W2, strides=(1, 2, 2, 1), padding="SAME", name="conv2")
        #cv1 = tf.nn.max_pool2d(cv1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME", name="maxpool1")

    with tf.name_scope("flatten"):
        x_target = cv1[context:]
        x_target = tf.reshape(x_target, shape=[1, dim2 ** 2])
        x_context = cv1[:context]
        x_context = tf.reshape(x_context, shape=[context, dim2 ** 2])

    with tf.name_scope("context_encoder"):
        dense1 = tf.matmul(x_context, Wco)
        dense1 = tf.reshape(dense1, shape=[1, dim2 ** 2])
        dense1 = tf.nn.bias_add(dense1, bias)

    with tf.name_scope("concat_outputs"):
        return tf.concat([dense1, x_target], axis=0)


def compute_euclidian_distance(x, y):
    return tf.reduce_sum(tf.square(x - y), 0)


def nn_loss(context, target):
    """euclid distance loss between context and target"""
    distance = compute_euclidian_distance(context, target)
    loss = tf.reduce_mean(tf.reduce_sum(distance))

    with tf.name_scope('add_l2_loss'):
        var = tf.trainable_variables()
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in var if "w" in v.name]) * 0.001
        loss += l2_loss

    return loss
