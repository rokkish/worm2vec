"""define Autoencoder"""
import torch
from torch import nn


class Autoencoder(nn.Module):
    
    def __init__(self, in_ch, out_ch, kernel_size):
        
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            
            nn.Conv2d(in_ch, out_ch, kernel_size),

            nn.BatchNorm2d(out_ch),

            nn.ReLU(inplace=True),

        )

        self.decoder = nn.Sequential(

            nn.Upsample(scale_factor=2)
        )

    def forward(self, x):

        x = self.encoder(x)

        x = self.decoder(x)
        
        return x
