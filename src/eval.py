

import os
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

import matplotlib.pyplot as plt
from skimage.io import imread
from PIL import Image
import argparse

import numpy as np

from models.vae import VAE
import train
import config


def evaluation(vae, eval_id):
    """ load datasets """
    train_loader, test_loader = train.load_datasets()

    if not os.path.exists('../results/' + eval_id):
        os.mkdir('../results/' + eval_id)

    vae.eval()

    sample_v = torch.randn(256, config.z_size).view(-1, config.z_size, 1, 1)

    for batch_idx, (data, target) in enumerate(train_loader):
        x = data
        x_rec, _, _ = vae.forward(x)
        resultsample = torch.cat([x, x_rec]) * 0.5 + 0.5
        resultsample = resultsample.cpu()
#        save_image(resultsample.view(-1, 1, config.IMG_SIZE, config.IMG_SIZE)[0],
#                    '../results/'+ eval_id +'/sample_encode.png')
        save_image(resultsample.view(-1, 1, config.IMG_SIZE, config.IMG_SIZE),
                    '../results/'+ eval_id +'/sample_encode.png')

        x_rec = vae.decode(sample_v)
        resultsample = x_rec * 0.5 + 0.5
        resultsample = resultsample.cpu()
        save_image(resultsample.view(-1, 1, config.IMG_SIZE, config.IMG_SIZE),
                    '../results/'+ eval_id +'/sample_decode.png')
        break

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--id", type=str, default="000")
    args = parse.parse_args()

    vae = VAE(zsize=config.z_size, layer_count=4, channels=1)
    vae.load_state_dict(torch.load("../models/VAEmodel.pkl"))

    evaluation(vae, args.id)
