"""predict mean of 2d-shapes
"""
import tensorflow as tf


class TwoDshape_model(object):
    def __init__(self, x_previous, x_next, x_now, input_dim):
        self.x_previous = x_previous
        self.x_next = x_next
        self.x_now = x_now
        self.input_dim = input_dim

        self.context = x_previous.shape[0] + x_next.shape[0]
        self.target = x_now.shape[0]
        self.N = self.context + self.target

        dim1 = input_dim//2
        dim2 = input_dim//4
        self.output_dim = dim2 ** 2 // 2 // 2

        w_initializer = tf.initializers.he_normal()
        b_initializer = tf.constant_initializer(1e-2)

        self.W_bn1 = self.createWeightsBN(input_dim)
        self.W_conv1 = tf.get_variable('w1', shape=[7, 7, input_dim, dim1], dtype=tf.float32, initializer=w_initializer)

        self.W_bn2 = self.createWeightsBN(dim1)
        self.W_conv2 = tf.get_variable('w2', shape=[7, 7, dim1, dim2], dtype=tf.float32, initializer=w_initializer)

        self.W_logit = tf.get_variable('w_logit1', shape=[self.output_dim, self.output_dim], dtype=tf.float32, initializer=w_initializer)
        self.bias_logit = tf.get_variable('b_logit1', shape=[self.output_dim], initializer=b_initializer)

    def nn(self):
        """Simple CNN model for prediction"""

        with tf.name_scope("concat_inputs"):
            x = tf.concat([self.x_previous, self.x_next, self.x_now], axis=0)
            x = tf.reshape(x, shape=[self.N, 1, self.input_dim, self.input_dim])

        with tf.name_scope("share_encoder"):
            cv1 = self.bn(x, self.W_bn1, name="bn1")
            cv1 = tf.nn.relu(cv1, name="relu1")
            cv1 = tf.nn.conv2d(cv1, self.W_conv1, strides=(1, 2, 2, 1), padding="SAME", name="conv1")
            cv1 = tf.nn.max_pool2d(cv1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME", name="maxpool1")

            cv2 = self.bn(cv1, self.W_bn2, name="bn2")
            cv2 = tf.nn.relu(cv2, name="relu2")
            cv2 = tf.nn.conv2d(cv2, self.W_conv2, strides=(1, 2, 2, 1), padding="SAME", name="conv2")
            cv2 = tf.nn.max_pool2d(cv2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME", name="maxpool2")

        with tf.name_scope("flatten"):
            x_target = cv2[self.context:]
            x_target = tf.reshape(x_target, shape=[self.target, self.output_dim])
            x_context = cv2[:self.context]
            x_context = tf.reshape(x_context, shape=[self.context, self.output_dim])

        with tf.name_scope("context_encoder"):
            dense1 = tf.matmul(x_context, self.W_logit)
            dense1 = tf.nn.bias_add(dense1, self.bias_logit)
            previous_context, next_context = dense1[:self.context//2], dense1[self.context//2:]
            dense1 = tf.math.add(previous_context, next_context)

        with tf.name_scope("concat_outputs"):
            return tf.concat([dense1, x_target], axis=0)

    def createWeightsBN(self, s):
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

    def bn(self, x, variables, name, eps=0.0001):
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
