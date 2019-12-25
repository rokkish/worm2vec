"""
    前処理
"""
import numpy as np
from skimage import color, filters, util
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import erosion, dilation
from PIL import Image

class ToBinary(object):
    def __init__(self):
        pass

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to binaryscale.

        Returns:
            PIL Image: Binaryscaled image.
        """
        img = np.asarray(img)
        thres = filters.threshold_otsu(img)
        img_bin = img > thres * 0.7
        img_bin = Image.fromarray(np.uint8(img_bin))
        return img_bin


class FillHole(object):
    def __init__(self):
        pass

    def __call__(self, img_bin):
        """
        Args:
            img (PIL Image): Hole of Image will be filled.

        Returns:
            PIL Image: Filled hole image.
        """
        img_bin = np.asarray(img_bin)

        img_fill_holes = util.invert(img_bin)
        img_fill_holes = binary_fill_holes(img_fill_holes).astype(int)
        img_fill_holes = util.invert(img_fill_holes)

        # erosion
        eroded = erosion(img_fill_holes)
        for i in range(3):
            eroded = erosion(eroded)

        # dilation
        dilated = dilation(eroded)
        for i in range(3):
            dilated = dilation(dilated)
        dilated = util.invert(dilated)
        dilated = binary_fill_holes(dilated).astype(int)

        dilated = Image.fromarray(np.uint8(dilated))
        return dilated

class ToNDarray(object):
    def __init__(self):
        pass

    def __call__(self, x):
        x = np.asarray(x)
        x_shape = x.shape    #x=(C,H,W)
        return x
