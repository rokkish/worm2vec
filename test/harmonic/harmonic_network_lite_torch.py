"""
Harmonic Convolutions Lite

A simplified API for harmomin_network_ops

pytorch version
"""

import tensorflow as tf
import torch
import torch.nn as nn
import harmonic_network_ops as hn_ops
import get_logger
logger = get_logger.get_logger(name='lite')


class HarmonicConv2d(nn.module):
    """Harmonic Convolution
    """
    def __init__(self, n_channels, ksize, strides=(1, 1, 1, 1), padding='VALID',
                 phase=True, max_order=1, stddev=0.4, n_rings=None):
        """
        Args:
            n_channels (int): number of output channels.
            ksize (int): size of square filter.
            strides (tuple, optional): stride size (4-tuple. Defaults to (1, 1, 1, 1).
            padding (str, optional): SAME or VALID. Defaults to 'VALID'.
            phase (bool, optional): use a per-channel phase offset. Defaults to True.
            max_order (int, optional): maximum rotation order e.g. max_order=2 uses 0,1,2. Defaults to 1.
            stddev (float, optional): scale of filter initialization wrt He initialization. Defaults to 0.4.
            n_rings ([type], optional): [description]. Defaults to None.
        """
        super().__init__()
        self.n_channels = n_channels
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        self.phase = phase
        self.max_order = max_order
        self.stddev = stddev
        self.n_rings = n_rings

    def forward(self, x):
        """
        Args:
            x ([type]): input tf tensor, shape [batchsize,height,width,order,complex,channels], e.g. a real input tensor of rotation order 0 could have shape [16,32,32,3,1,9], or a complex input tensor of rotation orders 0,1,2, could have shape [32,121,121,3,2,10].

        Returns:
            [type]: [description]
        """
        xsh = list(x.shape())
        shape = [self.ksize, self.ksize, xsh[5], self.n_channels]
        Q = hn_ops.get_weights_dict(shape, self.max_order, std_mult=self.stddev, n_rings=self.n_rings)

        if self.phase:
            P = hn_ops.get_phase_dict(xsh[5], self.n_channels, self.max_order)
        else:
            P = None

        self.W = hn_ops.get_filters(Q, filter_size=self.ksize, P=P, n_rings=self.n_rings)

        return hn_ops.h_conv(x, self.W, strides=self.strides,
                      padding=self.padding, max_order=self.max_order)


def batch_norm(x, train_phase, fnc=tf.nn.relu, decay=0.99, eps=1e-4, name='hbn'):
    """Batch normalization for the magnitudes of X"""
    return hn_ops.h_batch_norm(x, fnc, train_phase, decay=decay, eps=eps, name=name)


def non_linearity(x, fnc=tf.nn.relu, eps=1e-4, name='nl'):
    """Alter nonlinearity for the complex domains"""
    return hn_ops.h_nonlin(x, fnc, eps=eps, name=name)


def mean_pool(x, ksize=(1, 1, 1, 1), strides=(1, 1, 1, 1), name='mp'):
    """Mean pooling"""
    return hn_ops.mean_pooling(x, ksize=ksize, strides=strides)


def sum_magnitudes(x, eps=1e-12, keep_dims=True):
    """Sum the magnitudes of each of the complex feature maps in X.

    Output U = sum_i |x_i|

    x: input tf tensor, shape [batchsize,height,width,channels,complex,order],
    e.g. a real input tensor of rotation order 0 could have shape
    [16,32,32,3,1,1], or a complex input tensor of rotation orders 0,1,2, could
    have shape [32,121,121,32,2,3]
    eps: regularization since grad |x| is infinite at zero (default 1e-4)
    keep_dims: whether to collapse summed dimensions (default True)
    """
    R = tf.reduce_sum(tf.square(x), axis=[4], keep_dims=keep_dims)
    return tf.reduce_sum(tf.sqrt(tf.maximum(R, eps)), axis=[3], keep_dims=keep_dims)
