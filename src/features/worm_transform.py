"""
    前処理
"""
import numpy as np
from skimage import color, filters, util
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import erosion, dilation
from PIL import Image

# labelling
from skimage.measure import label, regionprops
from torchvision.transforms import functional

import get_logger
logger = get_logger.get_logger(name='transform')

class ToBinary(object):
    def __init__(self):
        pass

    def __call__(self, img):
        """
        Args:
            img (PIL Image) :   Image to be converted to binaryscale.

        Returns:
            img             :   np.array
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
            np.array image: labelled image. 最大領域画像の切り出し，オブジェクト内オブジェクトの消去
        """
        def get_largest_region(regions):
            tmp_area = 0
            for region in regions:
                tmp_area = max(region.area, tmp_area)
            for region in regions:
                if tmp_area == region.area:
                    return region

        def delete_obect_in_largest_region(img, largest_region, regions):

            lminr, lminc, lmaxr, lmaxc = largest_region.bbox

            for region in regions:

                if region != largest_region:

                    minr, minc, maxr, maxc = region.bbox

                    if minr > lminr and minc > lminc and maxr < lmaxr and maxc < lmaxc:
                        labeled_img = img[minr:maxr, minc:maxc]
                        img[minr:maxr, minc:maxc] = np.where(labeled_img > 0 , 0, labeled_img)

            return img

        def check_whether_on_edge(img, labeled_img, region):
            minr, minc, maxr, maxc = region.bbox

            # region.bbox is around edges.
            if minr < 10 or minc < 10 or maxr > img.shape[0] - 10 or maxc > img.shape[1] - 10:
                labeled_img = np.where(labeled_img > 0 , 0, labeled_img)

            return labeled_img

        # label image regions
        label_image = label(img)

        # take regions with largest enough areas
        regions = regionprops(label_image, coordinates='xy')
        region = get_largest_region(regions)

        img = delete_obect_in_largest_region(img, region, regions)

        # cut img
        minr, minc, maxr, maxc = region.bbox
        labeled_img = img[minr:maxr, minc:maxc]

        # detect on edge
        labeled_img = check_whether_on_edge(img, labeled_img, region)

        return labeled_img


class Padding(object):
    def __init__(self):
        pass

    def __call__(self, img):
        """
        Args:
            ndarray Image (W, H) :   labelled image.

        Returns:
            ndarray Image (W, H) :   Padded image.
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

        return img_padded


class Rotation(object):
    def __init__(self):
        pass

    def __call__(self, img):
        """
        Args:
            img (ndarray Image) (W, H) : Hole of Image will be filled.

        Returns:
            img (ndarray Image) (N, W, H) : Rotated image.
        """

        img = Image.fromarray(np.uint8(img), mode="L")
        w, h = img.size[0], img.size[1]

        rotated_img = np.reshape(np.asarray(img), (-1, w, h))

        for theta in range(10, 360, 10):

            img_ = functional.affine(img=img, angle=theta, translate=[1, 1], scale=1, shear=0, fillcolor=0)

            rotated_img = np.concatenate([rotated_img, np.reshape(np.asarray(img_), (-1, w, h))], 0)

        #print("rotated", rotated_img.shape)
        return rotated_img


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            ndarray Image (N, W, H) :   Rotated Image.

        Returns:
            ndarray Image (N, W, H) :   Resized image.
        """
        def resize_img(img):
            img = Image.fromarray(np.uint8(img), mode="L")
            img = functional.resize(img, self.size, self.interpolation)
            return np.asarray(img)

        for i, img_i in enumerate(img):
            img_ = resize_img(img_i)
            img_ = np.reshape(img_, (-1, img_.shape[0], img_.shape[1]))
            if i == 0:
                resized_img = img_
            else:
                resized_img = np.concatenate([resized_img, img_], 0)

        resized_img = np.reshape(resized_img, (-1, 1, resized_img.shape[1], resized_img.shape[2]))

        return resized_img


class Resize_nonbinary(object):
    def __init__(self, size, interpolation=Image.NEAREST):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            ndarray Image (N, W, H) :   Rotated Image.

        Returns:
            ndarray Image (N, W, H) :   Resized image.
        """
        def resize_img(img):
            img = Image.fromarray(np.uint8(img), mode="L")
            img = img.resize(self.size, self.interpolation)
            img = np.asarray(img)
            return img

        resized_img = np.zeros((img.shape[0], self.size[0], self.size[1]))

        for i in range(img.shape[0]):
            resized_img[i, :, :] = resize_img(img[i, :, :])
        return resized_img


class Normalize(object):
    def __init__(self, max_value=255):
        self.max_value = max_value
    def __call__(self, img):
        """
        Args:
            ndarray Image (N, W, H) :   

        Returns:
            ndarray Image (N, W, H) :   
        """

        return (img - img.min()) / (img.max() - img.min()) * self.max_value




class ToNDarray(object):
    def __init__(self):
        pass

    def __call__(self, x):
        x = np.asarray(x)
        return x
