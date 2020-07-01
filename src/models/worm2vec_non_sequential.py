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

class Worm2vec_nonseq(nn.Module):
    """this is neural net module"""
    def __init__(self, zsize, reverse):
        super(Worm2vec_nonseq, self).__init__()

        self.zsize = zsize
        self.inp_dim = 1
        self.reverse = reverse
        #self.loss_function = Lossfunction(loss_function_name, num_pos, batchsize, tau)
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

        self.classifier = nn.Sequential(
            nn.Linear(self.zsize, 4),
        )


    def encode(self, x):
        x = self.enc(x)
        x = torch.squeeze(x)
        return x

    def forward(self, x):
        enc_x = self.encode(x)
        enc_x = self.classifier(enc_x)
        return enc_x

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

def normal_init(m, mean, std):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class Lossfunction(nn.Module):
    def __init__(self, loss_function_name, num_pos, batchsize, tau):
        super(Lossfunction, self).__init__()

        self.loss_function_name = loss_function_name
        self.num_pos = num_pos
        self.batchsize = batchsize
        self.tau = tau

    def forward(self, x, labels):
        """return loss which is selected

        Args:
            x ([type]): [description]

        Raises:
            NameError: loss function name don't exist

        Returns:
            Loss[float]:
        """
        loss = []
        original_posneg_size = x.shape[0] // self.batchsize

        if self.loss_function_name == "NCE":

            for batch in range(self.batchsize):
                x_batch = x[batch * original_posneg_size: (batch + 1) * original_posneg_size]
                x_original, x_pos, x_neg = x_batch[0], x_batch[1: 1 + self.num_pos], x_batch[1 + self.num_pos:]

                for x_pos_i in x_pos:
                    cos_pos = self.exp_cos(x_original, x_pos_i)
                    cos_neg_list = [self.exp_cos(x_pos_i, x_neg_i) for x_neg_i in x_neg]
                    cos_neg = torch.sum(torch.FloatTensor(cos_neg_list))
                    loss.append(-math.log(cos_pos / (cos_pos + cos_neg)))

            loss = Variable(torch.FloatTensor(loss), requires_grad=True)
            return torch.mean(loss)

        elif self.loss_function_name == "TripletMargin":

            loss_func = nn.TripletMarginLoss(margin=1.0, p=2)
 
            for batch in range(self.batchsize):
                x_batch = x[batch * original_posneg_size: (batch + 1) * original_posneg_size]
                anchor, positive, negative = x_batch[0],\
                                             x_batch[1: 1 + self.num_pos],\
                                             x_batch[1 + self.num_pos: -1]
                loss.append(loss_func(anchor, positive, negative))
 
            loss = Variable(torch.FloatTensor(loss), requires_grad=True) 
            return torch.mean(loss)

        elif self.loss_function_name == "CrossEntropy":
            ce_fn = nn.CrossEntropyLoss()
            ce_loss = ce_fn(x, labels)
            return ce_loss
        else:
            raise NameError("No exist")

    @staticmethod
    def cosin(x1, x2):
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        return cos(x1, x2)

    def exp_cos(self, x1, x2):
        exp_cos = math.exp(self.cosin(x1, x2) / self.tau)
        return exp_cos

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
