"""class dataset"""

import os
import glob
from PIL import Image
import torch
import config

class WormDataset(torch.utils.data.Dataset):
    """
        Def Dataset
    """

    def __init__(self, root, train=True, transform=None):

        self.root = root    # root_dir \Tanimoto_eLife_Fig3B or \unpublished control
        self.train = train  # training set or test set
        self.transform = transform

        data_dirs_all = glob.glob(self.root + "/*")
        if self.train:
            #TODO:学習データ比決め内してるの修正．0.8
            data_dirs = data_dirs_all[:int(len(data_dirs_all) * 0.8)]
        else:
            data_dirs = data_dirs_all[int(len(data_dirs_all) * 0.8):]
        del data_dirs_all

        self.data = []
        for dir_i in data_dirs:
            self.data.extend(glob.glob(dir_i + "/main/*"))

            if len(self.data) > config.MAX_LEN_TRAIN_DATA:
                break

        #self.targets = self.data.copy()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where image == target.
        """
        img = self.data[index]
        #target = self.targets[index]

        img = Image.open(img)

        if self.transform is not None:
            img = self.transform(img)
            #target = self.transform(target)

        return img, img#, target

    def __len__(self):
        return len(self.data)
