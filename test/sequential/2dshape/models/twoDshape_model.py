"""predict mean of 2d-shapes
"""
import tensorflow as tf


def nn(x_previous, x_next, x_now, input_dim):
    """Simple CNN model for prediction"""

    context = x_previous.shape[0] + x_next.shape[0]
    target = x_now.shape[0]
    N = context + target

    dim1 = input_dim//2
    dim2 = input_dim//4
    output_dim = dim2 ** 2 // 2 // 2

    w_initializer = tf.initializers.he_normal()
    b_initializer = tf.constant_initializer(1e-2)

    W_bn1 = createWeightsBN(input_dim)
    W_conv1 = tf.get_variable('w1', shape=[N, 1, input_dim, dim1], dtype=tf.float32, initializer=w_initializer)

    W_bn2 = createWeightsBN(dim1)
    W_conv2 = tf.get_variable('w2', shape=[N, 1, dim1, dim2], dtype=tf.float32, initializer=w_initializer)

    W_logit = tf.get_variable('w_logit1', shape=[output_dim, output_dim], dtype=tf.float32, initializer=w_initializer)
    bias_logit = tf.get_variable('b_logit1', shape=[output_dim], initializer=b_initializer)

    with tf.name_scope("concat_inputs"):
        x = tf.concat([x_previous, x_next, x_now], axis=0)
        x = tf.reshape(x, shape=[N, 1, input_dim, input_dim])

    with tf.name_scope("share_encoder"):
        cv1 = bn(x, W_bn1, name="bn1")
        cv1 = tf.nn.relu(cv1, name="relu1")
        cv1 = tf.nn.conv2d(cv1, W_conv1, strides=(1, 2, 2, 1), padding="SAME", name="conv1")
        cv1 = tf.nn.max_pool2d(cv1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME", name="maxpool1")

        cv2 = bn(cv1, W_bn2, name="bn2")
        cv2 = tf.nn.relu(cv2, name="relu2")
        cv2 = tf.nn.conv2d(cv2, W_conv2, strides=(1, 2, 2, 1), padding="SAME", name="conv2")
        cv2 = tf.nn.max_pool2d(cv2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME", name="maxpool2")

    with tf.name_scope("flatten"):
        x_target = cv2[context:]
        x_target = tf.reshape(x_target, shape=[target, output_dim])
        x_context = cv2[:context]
        x_context = tf.reshape(x_context, shape=[context, output_dim])

    with tf.name_scope("context_encoder"):
        dense1 = tf.matmul(x_context, W_logit)
        dense1 = tf.nn.bias_add(dense1, bias_logit)
        previous_context, next_context = dense1[:context//2], dense1[context//2:]
        dense1 = tf.math.add(previous_context, next_context)

    with tf.name_scope("concat_outputs"):
        return tf.concat([dense1, x_target], axis=0)


def createWeightsBN(s):
    """
    Creates weights for batch normalization layer

    Parameters:
    -----------
    s: int
        size of to be normalized
    """
    gamma = tf.Variable(tf.truncated_normal([s]))
    beta = tf.Variable(tf.ones([s]))
    return [gamma, beta]


def bn(x, variables, name, eps=0.0001):
    """Applies Batch normalization
    from https://github.com/tmulc18/BatchNormalization/blob/master/Batch%20Normalization%20Convolution.ipynb
    """
    with tf.name_scope("batch_norm_" + name):
        gamma, beta = variables[0], variables[1]
        mu = tf.reduce_mean(x, keep_dims=True)
        sigma = tf.reduce_mean(tf.square(x - mu), keep_dims=True)
        x_hat = (x - mu) / tf.sqrt(sigma + eps)
    return gamma * x_hat + beta


def compute_euclidian_distance(x, y):
    return tf.reduce_sum(tf.square(x - y), 0)


def nn_loss(context, target):
    """euclid distance loss between context and target"""
    with tf.name_scope('euclid_distance_loss'):
        distance = compute_euclidian_distance(context, target)
        loss = tf.reduce_mean(tf.reduce_sum(distance))

    with tf.name_scope('add_l2_loss'):
        var = tf.trainable_variables()
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in var if "w" in v.name]) * 0.001
        loss += l2_loss

    return loss
