"""predict mean of 2d-shapes
"""
import tensorflow as tf


def nn(x, input_dim):
    """Simple AutoEncoder model"""

    bs = int(x.shape[0])
    input_size = int(x.shape[1])
    size1 = input_size // 2
    size2 = size1 // 2
    size3 = size2 // 2
    size4 = size3 // 2

    dim1 = input_dim * 2
    dim2 = input_dim * (2**2)
    dim3 = input_dim * (2**3)
    dim4 = input_dim * (2**4)

    strides = (1, 2, 2, 1)

    output_shape1 = [bs, size3, size3, dim3]
    output_shape2 = [bs, size2, size2, dim2]
    output_shape3 = [bs, size1, size1, dim1]
    output_shape4 = [bs, input_size, input_size, input_dim]

    w_initializer = tf.initializers.he_normal()

    W_bn1 = createWeightsBN(input_dim)
    W_bn2 = createWeightsBN(dim1)
    W_bn3 = createWeightsBN(dim2)
    W_bn4 = createWeightsBN(dim3)

    W_conv1 = tf.get_variable('w_c1', shape=[7, 7, input_dim, dim1], dtype=tf.float32, initializer=w_initializer)
    W_conv2 = tf.get_variable('w_c2', shape=[7, 7, dim1, dim2], dtype=tf.float32, initializer=w_initializer)
    W_conv3 = tf.get_variable('w_c3', shape=[7, 7, dim2, dim3], dtype=tf.float32, initializer=w_initializer)
    W_conv4 = tf.get_variable('w_c4', shape=[7, 7, dim3, dim4], dtype=tf.float32, initializer=w_initializer)

    W_deconv1 = tf.get_variable('w_dc1', shape=[7, 7, dim3, dim4], dtype=tf.float32, initializer=w_initializer)
    W_deconv2 = tf.get_variable('w_dc2', shape=[7, 7, dim2, dim3], dtype=tf.float32, initializer=w_initializer)
    W_deconv3 = tf.get_variable('w_dc3', shape=[7, 7, dim1, dim2], dtype=tf.float32, initializer=w_initializer)
    W_deconv4 = tf.get_variable('w_dc4', shape=[7, 7, input_dim, dim1], dtype=tf.float32, initializer=w_initializer)

    with tf.name_scope("reshape_inputs"):
        x = tf.reshape(x, shape=[bs, input_size, input_size, input_dim])

    with tf.name_scope("encoder"):
        enc = bn(x, W_bn1, name="bn1")
        enc = tf.nn.relu(enc, name="relu1")
        enc = tf.nn.conv2d(enc, W_conv1, strides=strides, padding="SAME", name="conv1")
        enc = tf.nn.max_pool2d(enc, ksize=(1, 2, 2, 1), strides=(1, 1, 1, 1), padding="SAME", name="maxpool1")

        enc = bn(enc, W_bn2, name="bn2")
        enc = tf.nn.relu(enc, name="relu2")
        enc = tf.nn.conv2d(enc, W_conv2, strides=strides, padding="SAME", name="conv2")
        enc = tf.nn.max_pool2d(enc, ksize=(1, 2, 2, 1), strides=(1, 1, 1, 1), padding="SAME", name="maxpool2")

        enc = bn(enc, W_bn3, name="bn3")
        enc = tf.nn.relu(enc, name="relu3")
        enc = tf.nn.conv2d(enc, W_conv3, strides=strides, padding="SAME", name="conv3")
        enc = tf.nn.max_pool2d(enc, ksize=(1, 2, 2, 1), strides=(1, 1, 1, 1), padding="SAME", name="maxpool3")

        enc = bn(enc, W_bn4, name="bn4")
        enc = tf.nn.relu(enc, name="relu4")
        enc = tf.nn.conv2d(enc, W_conv4, strides=strides, padding="SAME", name="conv4")
        enc = tf.nn.max_pool2d(enc, ksize=(1, 2, 2, 1), strides=(1, 1, 1, 1), padding="SAME", name="maxpool4")

    """
    with tf.name_scope("reshape"):
        enc = tf.reshape(cv4, shape=[bs, latent_dim])

    with tf.name_scope("reparametarizer"):
        z_mean = tf.matmul(cv4, W_reparam1)
        z_log_var = tf.matmul(cv4, W_reparam2)
        z = sampling(z_mean, z_log_var)
    with tf.name_scope("reshape"):
        z = tf.reshape(enc, shape=[bs, latent_dim])
    """

    with tf.name_scope("decoder"):
        dcv = tf.nn.conv2d_transpose(enc, W_deconv1, output_shape=output_shape1, strides=strides, padding="SAME", name="deconv1")
        dcv = tf.nn.conv2d_transpose(dcv, W_deconv2, output_shape=output_shape2, strides=strides, padding="SAME", name="deconv2")
        dcv = tf.nn.conv2d_transpose(dcv, W_deconv3, output_shape=output_shape3, strides=strides, padding="SAME", name="deconv3")
        dcv = tf.nn.conv2d_transpose(dcv, W_deconv4, output_shape=output_shape4, strides=strides, padding="SAME", name="deconv4")
        dcv = tf.nn.sigmoid(dcv, name="outlayer")

    with tf.name_scope("reshape3"):
        dec = tf.reshape(dcv, shape=[bs, input_size, input_size])
        return dec#, z_mean, z_log_var


def log_clip(value, min=1.e10, max=1.0):
    return tf.math.log(tf.clip_by_value(value, min, max))


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


def sampling(mean, log_var):
    """Use (mean, log_var) to sample z, the vector encoding shapes
    """
    with tf.name_scope("sampling"):
        batch, dim = tf.shape(mean)[0], tf.shape(mean)[1]
        stddev = tf.exp(0.5 * log_var)
        epsilon = tf.random.normal([batch, dim])
        return mean + stddev * epsilon


def kl_loss(mean, log_var):
    """KL divergence function
    """
    with tf.name_scope("kl_loss"):
        var = tf.exp(log_var)
        return -0.5 * tf.reduce_sum(
            log_var - tf.square(mean) - var + 1
        )


def compute_binary_cross_entropy(x, y):
    return tf.reduce_mean(x * log_clip(y) + (1 - x) * log_clip(1 - y))


def nn_euclid_loss(x, recon_x):
    with tf.name_scope('mean_squared_error'):
        return tf.reduce_mean((x - recon_x)**2)


def nn_loss(x, recon_x, z_mean, z_log_var):
    """euclid distance loss between context and target"""
    with tf.name_scope('cross_entropy_loss'):
        reconstruction = compute_binary_cross_entropy(x, recon_x)
        loss = reconstruction

    with tf.name_scope('add_l2_loss'):
        var = tf.trainable_variables()
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in var if "w" in v.name]) * 0.001
        loss += l2_loss

    with tf.name_scope('add_kl_loss'):
        kl = kl_loss(z_mean, z_log_var)
        loss += kl

    return loss
