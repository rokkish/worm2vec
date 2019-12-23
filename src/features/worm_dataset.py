"""class dataset"""

import os
import glob
from PIL import Image
import torch

class WormDataset(torch.utils.data.Dataset):
    """
        Def Dataset
        TODO:root内dir全て読み込めるようにする．
    """

    training_dir = '201302081337/main'
    test_dir = '201302081353/main'

    def __init__(self, root, train=True, transform=None):

        self.root = root    # root_dir \Tanimoto_eLife_Fig3B or \unpublished control
        self.train = train  # training set or test set
        self.transform = transform

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        if self.train:
            data_dir = self.training_dir
        else:
            data_dir = self.test_dir

        self.data = glob.glob(self.root + data_dir + "/*")
        self.targets = self.data.copy()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where image == target.
        """
        img, target = self.data[index], self.targets[index]

        img = Image.open(img)
        target = Image.open(target)

        if self.transform is not None:
            img = self.transform(img)
            target = self.transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


    def _check_exists(self):
        print(self.root + self.training_dir)
        return (os.path.exists(self.root + self.training_dir) and
                os.path.exists(self.root + self.test_dir))
