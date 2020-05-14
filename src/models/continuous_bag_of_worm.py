"""Continuous bag-of-worm model
    input [t0, t2], and predict [t1]
"""

import torch
import torch.nn as nn
import get_logger
logger = get_logger.get_logger(name='cbow')


class CBOW(nn.Module):
    """this is neural net module"""
    def __init__(self, zsize, loss_function_name):
        super(CBOW, self).__init__()

        self.zsize = zsize
        self.loss_function_name = loss_function_name
        self.inp_dim = 1
        #TODO:if zsize < 2**5, error happen!
        self.mod_dim1 = self.zsize//(2**5)
        self.mod_dim2 = self.mod_dim1*2
        self.mod_dim3 = self.mod_dim2*2
        self.mod_dim4 = self.mod_dim3*2
        self.mod_dim5 = self.mod_dim4*2
        self.mod_dim6 = self.mod_dim5*2

        self.enc = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(self.inp_dim, self.mod_dim1, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(self.mod_dim1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Conv2d(self.mod_dim1, self.mod_dim2, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(self.mod_dim2),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Conv2d(self.mod_dim2, self.mod_dim3, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(self.mod_dim3),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Conv2d(self.mod_dim3, self.mod_dim4, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(self.mod_dim4),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Conv2d(self.mod_dim4, self.mod_dim5, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(self.mod_dim5),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

        )

    def encode(self, x):
        x = self.enc(x)
        x = torch.squeeze(x)
        return x

    def multi_encode(self, x):
        left_x, right_x = x[0], x[1]
        left_x = self.encode(left_x)
        right_x = self.encode(right_x)
        return left_x, right_x

    def cat_latent_vector(self, left_z, right_z):
        z = torch.mean(left_z + right_z, 0, True)
        return z

    def forward(self, x, y):
        """"TODO:encode並列化可能?, x:gpu0, y:gpu1"""
        left_z, right_z = self.multi_encode(x)
        z = self.cat_latent_vector(left_z, right_z)
        y = self.encode(y)
        return self.loss_function(z, y, self.loss_function_name)

    @staticmethod
    def loss_function(x, y, loss_function_name):
        if loss_function_name == "binarycrossentropyLoss":
            return torch.mean((y - x)**2)
        else:
            raise NameError("No exist")


"""TEST
from models.continuous_bag_of_worm import CBOW
model = CBOW(zsize=64, loss_function_name="binarycrossentropyLoss")
import torch
t1 = torch.randn(2, 200, 1, 64, 64)
t2 = torch.randn(200, 1, 64, 64)
device = torch.device("cuda:1")
model, t1, t2 = model.to(device), t1.to(device), t2.to(device)
loss = model.forward(t1, t2)

x.shape
# seq       :  torch.Size([1, 128, 1, 1])
# squeeze   :  torch.Size([128])
"""
