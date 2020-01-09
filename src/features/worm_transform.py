"""
    前処理
"""
import numpy as np
from skimage import color, filters, util
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import erosion, dilation
from PIL import Image
import config

# labelling
import matplotlib.patches as mpatches
from skimage.measure import label, regionprops

class ToBinary(object):
    def __init__(self):
        pass

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to binaryscale.

        Returns:
            #PIL Image: Binaryscaled image.
            img:np.array
        """
        img = np.asarray(img)
        thres = filters.threshold_otsu(img)
        img_bin = img > thres #* 0.7
        #img_bin = Image.fromarray(np.uint8(img_bin))
        return img_bin


class FillHole(object):
    def __init__(self):
        pass

    def __call__(self, img_bin):
        """
        Args:
            img (PIL Image): Hole of Image will be filled.

        Returns:
            np.array image: Filled
        """
        #img_bin = np.asarray(img_bin)

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
        #dilated = Image.fromarray(np.uint8(dilated))
        return dilated

class Labelling(object):
    def __init__(self):
        pass

    def __call__(self, img):
        """
        Args:
            img (ndarray Image): Hole of Image will be filled.

        Returns:
            np.array image: labelled image.
        """
        def get_largest_region(regions):
            tmp_area = 0
            for region in regions:
                tmp_area = max(region.area, tmp_area)
            for region in regions:
                if tmp_area == region.area:
                    return region

        # label image regions
        label_image = label(img)

        # take regions with largest enough areas
        regions = regionprops(label_image)
        region = get_largest_region(regions)

        # show img
        minr, minc, maxr, maxc = region.bbox
        labeled_img = img[minr:maxr, minc:maxc]
        return labeled_img

class Padding(object):
    def __init__(self):
        pass

    def __call__(self, img):
        """
        Args:
            img (ndarray Image): Hole of Image will be filled.

        Returns:
            PIL Image: padded image.
        """

        x_width = max((450 - img.shape[0]) // 2, 0)
        y_width = max((450 - img.shape[1]) // 2, 0)
        if x_width < 0 or y_width < 0:
            print(x_width, y_width)
            raise ValueError("width over 400")

        x_width_up = x_width
        y_width_up = y_width
        if img.shape[0] % 2 == 1:
            x_width_up += 1
        if img.shape[1] % 2 == 1:
            y_width_up += 1
        img_padded = util.pad(img, [(x_width_up, x_width), (y_width_up, y_width)], mode="constant")

        img_padded = Image.fromarray(np.uint8(img_padded))

        return img_padded

class ToNDarray(object):
    def __init__(self):
        pass

    def __call__(self, x):
        x = np.asarray(x)
        x_shape = x.shape    #x=(C,H,W)
        return x
