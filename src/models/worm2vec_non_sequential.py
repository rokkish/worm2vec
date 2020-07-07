"""Worm to vector model at one time with negative/positive samples.
Input
    [original, positive*3, negative*4]
Output
    encode (N*8, C, H, W) into (N*8, Z)
"""

import torch
import torch.nn as nn
import get_logger
logger = get_logger.get_logger(name='worm2vec_nonseq')
import math
from torch.autograd import Variable
#from pytorch_metric_learning import losses
import config

class Worm2vec_nonseq(nn.Module):
    """this is neural net module"""
    def __init__(self, zsize):
        super(Worm2vec_nonseq, self).__init__()

        self.zsize = zsize
        self.inp_dim = 1
        #TODO:if zsize < 2**5, error happen!
        self.mod_dim1 = self.zsize//(2**4)
        self.mod_dim2 = self.mod_dim1*2
        self.mod_dim3 = self.mod_dim2*2
        self.mod_dim4 = self.mod_dim3*2
        self.mod_dim5 = self.mod_dim4*2

        self.enc = nn.Sequential(
            nn.MaxPool2d(2),

            nn.Conv2d(self.inp_dim, self.mod_dim1, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(self.mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.mod_dim1, self.mod_dim1, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(self.mod_dim1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(self.mod_dim1, self.mod_dim2, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(self.mod_dim2),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.mod_dim2, self.mod_dim2, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(self.mod_dim2),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(self.mod_dim2, self.mod_dim3, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(self.mod_dim3),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.mod_dim3, self.mod_dim3, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(self.mod_dim3),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(self.mod_dim3, self.mod_dim4, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(self.mod_dim4),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.mod_dim4, self.mod_dim4, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(self.mod_dim4),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(self.mod_dim4, self.mod_dim5, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(self.mod_dim5),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.mod_dim5, self.mod_dim5, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(self.mod_dim5),
            nn.ReLU(inplace=True),

            nn.AvgPool2d(2),

        )
        self.m_original = nn.Linear(self.zsize, self.zsize//2)
#        self.m_positive = nn.Linear(self.zsize, self.zsize//2)

    def encode(self, x):
        x = self.enc(x)
        x = torch.squeeze(x)
        return x

    def forward(self, anc, pos, neg):
        """

        Args:
            anc (tensor): anchor: (Batch * (1 ~ 3), C, H, W)

        Returns:
            tensor: (Batch * R, Z)
        """
        if len(anc.shape) != 4:
            raise ValueError("tensor shape not matched")
        anc_embedding = self.encode(anc)
        pos_embedding = self.encode(pos)
        neg_embedding = self.encode(neg)

        anc_embedding = self.m_original(anc_embedding)
        pos_embedding = self.m_original(pos_embedding)
        neg_embedding = self.m_original(neg_embedding)

        if len(anc_embedding.shape) != len(pos_embedding) and len(anc_embedding.shape) == 1:
            anc_embedding = anc_embedding.view(1, anc_embedding.shape[0])

        #logger.debug("shape:{}{}{}".format(anc_embedding.shape, pos_embedding.shape, neg_embedding.shape))

        #cat = torch.cat([anc_embedding, pos_embedding, neg_embedding])

        return {"anc_embedding": anc_embedding, "pos_embedding": pos_embedding, "neg_embedding": neg_embedding}

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

def normal_init(m, mean, std):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class Lossfunction(nn.Module):
    def __init__(self, loss_function_name, num_pos, num_neg, batchsize, tau, margin=config.MARGIN_TRIPLET):
        super(Lossfunction, self).__init__()

        self.loss_function_name = loss_function_name
        self.num_pos = num_pos
        self.num_neg = num_neg 
        self.batchsize = batchsize
        self.tau = tau
        self.margin = margin
        self.posneg_idx_combi_list = [[i_anc, i_pos, i_neg] for i_anc in range(self.batchsize) for i_pos in range(self.num_pos) for i_neg in range(self.num_neg)]

    def forward(self, x):
        """return loss which is selected

        Args:
            x (dic:tensor): {name:(Batch * R, z)}

        Raises:
            NameError: loss function name don't exist

        Returns:
            Loss[float]:
        """
        loss = []

        if self.loss_function_name == "TripletMargin":
            loss_func = nn.TripletMarginLoss(margin=self.margin, p=2)

            anc_embedding = x["anc_embedding"]
            pos_embedding = x["pos_embedding"]
            neg_embedding = x["neg_embedding"]
            #logger.debug("anc:{}".format(anc_embedding.shape))

            for (batch, i_pos, i_neg) in self.posneg_idx_combi_list:
                if batch >= anc_embedding.shape[0]:
                    logger.debug("batch{}, anc:{}".format(batch, anc_embedding.shape))
                    break
                anc = anc_embedding[batch].unsqueeze(0)
                pos = pos_embedding[batch * self.num_pos + i_pos].unsqueeze(0)
                neg = neg_embedding[batch * self.num_neg + i_neg].unsqueeze(0)

                loss.append(loss_func(anc, pos, neg))

            loss = Variable(torch.FloatTensor(loss), requires_grad=True) 

            return torch.mean(loss)

        else:
            raise NameError("No exist")

"""TEST
from models.worm2vec_non_sequential import Worm2vec_nonseq
model = Worm2vec_nonseq(zsize=32, loss_function_name="NCE", num_pos=3, batchsize=1, tau=1)
import torch
t1 = torch.randn(8, 1, 64, 64)
device = torch.device("cuda:1")
model, t1 = model.to(device), t1.to(device)
loss = model.forward(t1)

x.shape
# seq       :  torch.Size([1, 128, 1, 1])
# squeeze   :  torch.Size([128])
"""
