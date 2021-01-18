"""predict mean of 2d-shapes
"""
import tensorflow as tf


class AutoEncoder(object):
    def __init__(self, x, input_dim, multiply_dim):

        self.x = x
        self.input_dim = input_dim

        self.bs = int(self.x.shape[0])
        self.input_size = int(self.x.shape[1])

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

        self.output_shape1 = [self.bs, size3, size3, dim3]
        self.output_shape2 = [self.bs, size2, size2, dim2]
        self.output_shape3 = [self.bs, size1, size1, dim1]
        self.output_shape4 = [self.bs, self.input_size, self.input_size, self.input_dim]

        w_initializer = tf.initializers.he_normal()

        self.W_bn1 = self.createWeightsBN(input_dim)
        self.W_bn2 = self.createWeightsBN(dim1)
        self.W_bn3 = self.createWeightsBN(dim2)
        self.W_bn4 = self.createWeightsBN(dim3)

        self.W_conv1 = tf.get_variable('w_c1', shape=[7, 7, input_dim, dim1], dtype=tf.float32, initializer=w_initializer)
        self.W_conv2 = tf.get_variable('w_c2', shape=[7, 7, dim1, dim2], dtype=tf.float32, initializer=w_initializer)
        self.W_conv3 = tf.get_variable('w_c3', shape=[7, 7, dim2, dim3], dtype=tf.float32, initializer=w_initializer)
        self.W_conv4 = tf.get_variable('w_c4', shape=[7, 7, dim3, dim4], dtype=tf.float32, initializer=w_initializer)

        self.W_deconv1 = tf.get_variable('w_dc1', shape=[7, 7, dim3, dim4], dtype=tf.float32, initializer=w_initializer)
        self.W_deconv2 = tf.get_variable('w_dc2', shape=[7, 7, dim2, dim3], dtype=tf.float32, initializer=w_initializer)
        self.W_deconv3 = tf.get_variable('w_dc3', shape=[7, 7, dim1, dim2], dtype=tf.float32, initializer=w_initializer)
        self.W_deconv4 = tf.get_variable('w_dc4', shape=[7, 7, input_dim, dim1], dtype=tf.float32, initializer=w_initializer)

    def nn(self):
        """Simple AutoEncoder model"""

        with tf.name_scope("reshape_inputs"):
            x = tf.reshape(self.x, shape=[self.bs, self.input_size, self.input_size, self.input_dim])

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

        with tf.name_scope("decoder"):
            dcv = tf.nn.conv2d_transpose(enc, self.W_deconv1, output_shape=self.output_shape1, strides=self.strides, padding="SAME", name="deconv1")
            dcv = tf.nn.conv2d_transpose(dcv, self.W_deconv2, output_shape=self.output_shape2, strides=self.strides, padding="SAME", name="deconv2")
            dcv = tf.nn.conv2d_transpose(dcv, self.W_deconv3, output_shape=self.output_shape3, strides=self.strides, padding="SAME", name="deconv3")
            dcv = tf.nn.conv2d_transpose(dcv, self.W_deconv4, output_shape=self.output_shape4, strides=self.strides, padding="SAME", name="deconv4")
            dcv = tf.nn.sigmoid(dcv, name="outlayer")

        with tf.name_scope("reshape3"):
            dec = tf.reshape(dcv, shape=[self.bs, self.input_size, self.input_size])
            return dec#, z_mean, z_log_var

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


def nn_euclid_loss(x, recon_x):
    with tf.name_scope('mean_squared_error'):
        return tf.reduce_mean((x - recon_x)**2)
