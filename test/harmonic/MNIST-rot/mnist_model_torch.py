"""
Harmonic Convolution
"""
import torch
import torch.nn as nn
import torch.optim as optim
import get_logger
logger = get_logger.get_logger(name='train')

from harmonic_network_lite_torch import HarmonicConv2d
import harmonic_network_lite_torch as hn_lite


class HarmonicNet(nn.module):
    def __init__(self, args):
        super(HarmonicNet, self).__init__()

        # Number of Filters
        nf = args.n_filters
        nf2 = int(nf*args.filter_gain)
        nf3 = int(nf*(args.filter_gain**2.))
        bs = args.batch_size
        fs = args.filter_size
        ncl = args.n_classes
        sm = args.std_mult
        nr = args.n_rings

        self.non_linearity = hn_lite.non_linearity()
        self.batch_norm = hn_lite.batch_norm()
        self.mean_pool = hn_lite.mean_pool()
        self.sum_magnitudes = hn_lite.sum_magnitudes(),

        self.dualconv_block1 = DualHConv2d(nf, fs, nr)
        self.dualconv_block2 = DualHConv2d(nf2, fs, nr)
        self.dualconv_block3 = DualHConv2d(nf3, fs, nr)
        self.hconv2d_block4 = HarmonicConv2d(n_channels=ncl, ksize=fs, padding="SAME", n_rings=nr)

        self.enc = nn.Sequential(
            self.dualconv_block1,

            self.mean_pool,

            self.dualconv_block2,

            self.mean_pool,

            self.dualconv_block3,

            self.hconv2d_block4,

            self.sum_magnitudes,

            #cv7 = tf.reduce_mean(real, axis=[1,2,3,4])
            #tf.nn.bias_add(cv7, bias)
        )

    def forward(self, x):
        x = self.enc(x)
        return x


class DualHConv2d(nn.module):
    def __init__(self, nf, fs, nr):
        self.hconv2d_blockn = HarmonicConv2d(n_channels=nf, ksize=fs, padding="SAME", n_rings=nr)
        self.dualconv_blockn = nn.Sequential(
            self.hconv2d_blockn,
            self.non_linearity,
            self.hconv2d_blockn,
            self.batch_norm,
        )

    def forward(self, x):
        return self.dualconv_blockn(x)
