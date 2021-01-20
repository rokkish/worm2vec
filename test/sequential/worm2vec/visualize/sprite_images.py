import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_sprite_image(images):
    """create image for tensorboard projector.

    Args:
        images (tensor): (batch size, H, W)
    """

    # size of sprite image (28 pixel)
    img_h = images.shape[1]
    img_w = images.shape[2]

    # length of sprite imae
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    # init sprite image
    sprite_image = np.zeros((img_h * n_plots, img_w * n_plots))

    for i in range(n_plots):
        for j in range(n_plots):
            filter_ij = i * n_plots + j

            if filter_ij < images.shape[0]:
                sprite_image[i * img_h: (i + 1) * img_h, j * img_w: (j + 1) * img_w] \
                    = np.reshape(images[filter_ij], (img_h, img_w))
    return sprite_image


def save_sprite_image(sprite_image, path):
    plt.imsave(path, sprite_image, cmap="Greys_r")
    plt.close()
