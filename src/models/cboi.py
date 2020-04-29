"""
    Continuously Bag Of Images
"""

import torch
from torch import nn
from torch.nn import functional as F

class CBOI(nn.Module):
    def __init__(self, params):
        self.params = params

    def forward(self, contexts, target):
        loss = 0
        return loss

    def backward(self, dout=1):

        return None
