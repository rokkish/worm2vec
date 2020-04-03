"""class dataset"""

import os
import glob
from PIL import Image
import torch
import config
import get_logger
logger = get_logger.get_logger(name='dataset')

class WormDataset(torch.utils.data.Dataset):
    """
        Def Dataset
    """

    def __init__(self, root, train=True, transform=None, window=3):

        self.root = root    # root_dir \Tanimoto_eLife_Fig3B or \unpublished control
        self.train = train  # training set or test set
        self.transform = transform
        self.window = window
        self.data = []

        tensor_all = glob.glob(self.root + "/*")
        if self.train:
            self.data.extend(tensor_all[:int(len(tensor_all) * 0.8)])
        else:
            self.data.extend(tensor_all[int(len(tensor_all) * 0.8):])

        if len(self.data) > config.MAX_LEN_TRAIN_DATA:
            self.data = self.data[:config.MAX_LEN_TRAIN_DATA]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (context, target, OutOfRange)
                context     : tensor(N, R, C, H, W)
                target      : tensor(1, R, C, H, W)

        """

        if index - self.window < 0 or index + 1 + self.window > len(self.data):
            dummy_path = self.data[index]
            dummy = torch.load(dummy_path).type(torch.float)
            return dummy

        target_path = self.data[index]
        left_context_path = self.data[index - self.window:index]
        right_context_path = self.data[index + 1:index + 1 + self.window]

        target = torch.load(target_path)
        target = target.type(torch.float)
        target = target.unsqueeze(0)
        context = self.load_tensor(left_context_path + right_context_path)
        context = context.type(torch.float)

        return torch.cat([target, context], dim=0)

    def __len__(self):
        return len(self.data)

    def load_tensor(self, pathList):
        for i, path in enumerate(pathList):
            tmp = torch.load(path)
            tmp = tmp.unsqueeze(0)
            if i == 0:
                tensors = tmp.clone()
            else:
                tensors = torch.cat([tensors, tmp], dim=0)
        return tensors
