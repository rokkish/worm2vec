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

    def __init__(self, root, train=True, transform=None, processed=False, window=3):

        self.root = root    # root_dir \Tanimoto_eLife_Fig3B or \unpublished control
        self.train = train  # training set or test set
        self.transform = transform
        self.processed = processed # Use processed data ? or raw data
        self.window = window
        self.data = []

        if self.processed:
            tensor_all = glob.glob(self.root + "/*")
            if self.train:
                self.data.extend(tensor_all[:int(len(tensor_all) * 0.8)])
            else:
                self.data.extend(tensor_all[int(len(tensor_all) * 0.8):])

            if len(self.data) > config.MAX_LEN_TRAIN_DATA:
                self.data = self.data[:config.MAX_LEN_TRAIN_DATA]

        else:
            data_dirs_all = glob.glob(self.root + "/*")
            if self.train:
                #TODO:学習データ比決め内してるの修正．0.8
                data_dirs = data_dirs_all[:int(len(data_dirs_all) * 0.8)]
            else:
                data_dirs = data_dirs_all[int(len(data_dirs_all) * 0.8):]

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

        if self.processed == True:
            t = torch.load(img)
            img = t[0].type(torch.float)
            target = t[1:].type(torch.float)
        else:
            img = Image.open(img)

        if self.transform is not None:
            pass
            #img = self.transform(img)
            #target = self.transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def load_tensor(self, pathList):
        for i, path in enumerate(pathList):
            tmp = torch.load(path)
            if i == 0:
                tensors = torch.load(pathList[i]).unsqueeze(0)
            else:
                tensors = torch.cat([tensors, tmp.unsqueeze(0)], dim=0)
        return tensors
