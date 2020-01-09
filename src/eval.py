

import os
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt
from skimage.io import imread
from PIL import Image
import argparse

import numpy as np

# 自作
from models.vae import VAE
from visualization.save_images_gray_grid import save_images_grid
import train
import config

import matplotlib.pyplot as plt

def plot_x(x, id_):
    plt.imsave(r"c:\Users\u853337i\Desktop\worm2vec\worm2vec/results/eval_x_" + id_ + ".jpg", x, cmap="gray")

def evaluation(vae, eval_id):
    """ load datasets """
    train_loader, test_loader = train.load_datasets()

    if not os.path.exists(r'c:\Users\u853337i\Desktop\worm2vec\worm2vec/results/' + eval_id):
        os.mkdir(r'c:\Users\u853337i\Desktop\worm2vec\worm2vec/results/' + eval_id)

    vae.eval()

    sample_v = torch.randn(256, config.z_size).view(-1, config.z_size, 1, 1)

    for batch_idx, (data, target) in enumerate(train_loader):
        x = data
        x_rec, _, _ = vae.forward(x)
        x_rec = (x_rec - x_rec.min()) / (x_rec.max() - x_rec.min())
        x = (x - x.min()) / (x.max() - x.min())
        resultsample = torch.cat([x, x_rec]) 
        resultsample = resultsample.cpu()

        plot_x(x[0, 0], "x0")
        plot_x(resultsample[0, 0].detach().numpy(), "r0")
        plot_x(resultsample[1, 0].detach().numpy(), "r1")

        #save_image(resultsample.view(-1, 1, config.IMG_SIZE, config.IMG_SIZE),
        save_images_grid(resultsample, nrow=16, scale_each=True,
                    filename=r'c:\Users\u853337i\Desktop\worm2vec\worm2vec/results/'+ eval_id +'/sample_encode.png')

        x_rec = vae.decode(sample_v)
        resultsample = x_rec * 0.5 + 0.5
        resultsample = resultsample.cpu()
        save_images_grid(resultsample, nrow=16, scale_each=True,
                    filename=r'c:\Users\u853337i\Desktop\worm2vec\worm2vec/results/'+ eval_id +'/sample_decode.png')

        break

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--id", type=str, default="000")
    args = parse.parse_args()

    vae = VAE(zsize=config.z_size, layer_count=config.layer_count, channels=1)
    vae.load_state_dict(torch.load(r"C:\Users\u853337i\Desktop\worm2vec\worm2vec/models/VAEmodel.pkl"))

    evaluation(vae, args.id)
