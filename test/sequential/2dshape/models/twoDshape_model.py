"""predict mean of 2d-shapes
"""
import tensorflow as tf
from models.autoencoder import AutoEncoder


class TwoDshape_model(object):
    def __init__(self, x_previous, x_next, x_now, input_dim, multiply_dim):
        self.x_previous = x_previous
        self.x_next = x_next
        self.x_now = x_now
        self.input_dim = input_dim

        self.context = x_previous.shape[0] + x_next.shape[0]
        self.target = x_now.shape[0]
        self.N = self.context + self.target
        self.input_size = int(self.x_now.shape[1])

        size1 = self.input_size // 2
        size2 = size1 // 2
        size3 = size2 // 2
        size4 = size3 // 2

        multiply_dim = list(map(int, multiply_dim.split("-")))
        dim1 = input_dim * multiply_dim[0]
        dim2 = input_dim * multiply_dim[1]
        dim3 = input_dim * multiply_dim[2]
        dim4 = input_dim * multiply_dim[3]

        self.strides = (1, 2, 2, 1)

        # share encoder output shape (N, size4, size4, dim4) -> (N, output_dim)
        self.share_output_dim = size4 ** 2 * dim4
        self.output_dim = self.share_output_dim

        w_initializer = tf.initializers.he_normal()
        b_initializer = tf.constant_initializer(1e-2)

        self.W_bn1 = self.createWeightsBN(input_dim)
        self.W_bn2 = self.createWeightsBN(dim1)
        self.W_bn3 = self.createWeightsBN(dim2)
        self.W_bn4 = self.createWeightsBN(dim3)

        self.W_conv1 = tf.get_variable('w_c1', shape=[7, 7, input_dim, dim1], dtype=tf.float32, initializer=w_initializer)
        self.W_conv2 = tf.get_variable('w_c2', shape=[7, 7, dim1, dim2], dtype=tf.float32, initializer=w_initializer)
        self.W_conv3 = tf.get_variable('w_c3', shape=[7, 7, dim2, dim3], dtype=tf.float32, initializer=w_initializer)
        self.W_conv4 = tf.get_variable('w_c4', shape=[7, 7, dim3, dim4], dtype=tf.float32, initializer=w_initializer)

        self.W_logit = tf.get_variable('w_logit1', shape=[self.share_output_dim, self.output_dim], dtype=tf.float32, initializer=w_initializer)
        self.bias_logit = tf.get_variable('b_logit1', shape=[self.output_dim], initializer=b_initializer)

    def nn(self):
        """Simple CNN model for prediction"""

        with tf.name_scope("concat_inputs"):
            x = tf.concat([self.x_previous, self.x_next, self.x_now], axis=0)
            x = tf.reshape(x, shape=[self.N, self.input_size, self.input_size, self.input_dim])

        x_enc = self.share_encoder(x)

        with tf.name_scope("flatten"):
            x_target = x_enc[self.context:]
            x_target = tf.reshape(x_target, shape=[self.target, self.output_dim])
            x_context = x_enc[:self.context]
            x_context = tf.reshape(x_context, shape=[self.context, self.output_dim])

        with tf.name_scope("context_encoder"):
            previous_context, next_context = x_context[:self.context//2], x_context[self.context//2:]
            dense1 = tf.math.add(previous_context, next_context)
            dense1 = tf.matmul(dense1, self.W_logit)
            dense1 = tf.nn.bias_add(dense1, self.bias_logit)

        with tf.name_scope("concat_outputs"):
            return tf.concat([dense1, x_target], axis=0)

    def share_encoder(self, x):
        """encoder of autoencoder.py
        """
        with tf.name_scope("encoder"):
            enc = self.bn(x, self.W_bn1, name="bn1")
            enc = tf.nn.relu(enc, name="relu1")
            enc = tf.nn.conv2d(enc, self.W_conv1, strides=self.strides, padding="SAME", name="conv1")
            enc = tf.nn.max_pool2d(enc, ksize=(1, 2, 2, 1), strides=(1, 1, 1, 1), padding="SAME", name="maxpool1")

            enc = self.bn(enc, self.W_bn2, name="bn2")
            enc = tf.nn.relu(enc, name="relu2")
            enc = tf.nn.conv2d(enc, self.W_conv2, strides=self.strides, padding="SAME", name="conv2")
            enc = tf.nn.max_pool2d(enc, ksize=(1, 2, 2, 1), strides=(1, 1, 1, 1), padding="SAME", name="maxpool2")

            enc = self.bn(enc, self.W_bn3, name="bn3")
            enc = tf.nn.relu(enc, name="relu3")
            enc = tf.nn.conv2d(enc, self.W_conv3, strides=self.strides, padding="SAME", name="conv3")
            enc = tf.nn.max_pool2d(enc, ksize=(1, 2, 2, 1), strides=(1, 1, 1, 1), padding="SAME", name="maxpool3")

            enc = self.bn(enc, self.W_bn4, name="bn4")
            enc = tf.nn.relu(enc, name="relu4")
            enc = tf.nn.conv2d(enc, self.W_conv4, strides=self.strides, padding="SAME", name="conv4")
            enc = tf.nn.max_pool2d(enc, ksize=(1, 2, 2, 1), strides=(1, 1, 1, 1), padding="SAME", name="maxpool4")

        return enc

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
