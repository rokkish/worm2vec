"""class dataset"""

import glob
import torch
import torch.utils.data
import config
import get_logger
import numpy as np
logger = get_logger.get_logger(name='dataset')
from features.sort_index import get_binaryfile_number


class WormDataset(torch.utils.data.Dataset):
    """
        Def Dataset
    """

    def __init__(self, root, train=True, transform=None, window=3, sequential=True):

        self.root = root    # root_dir \Tanimoto_eLife_Fig3B or \unpublished control
        self.train = train  # training set or test set
        self.transform = transform
        self.window = window
        self.sequential = sequential
        self.data = []
        self.data_index = 0
        self.count_skip_data = 0

        tensor_all = glob.glob(self.root + "/*")
        tensor_all.sort(key=get_binaryfile_number)
        if self.train:
            self.data.extend(tensor_all[:int(len(tensor_all) * 0.8)])
        else:
            self.data.extend(tensor_all[int(len(tensor_all) * 0.8):])
            if config.BATCH_SIZE == 1:
                self.data = self.data[:config.MAX_LEN_TRAIN_DATA]
            else:
                self.data = self.data[:-(len(self.data) % config.BATCH_SIZE)] # BATCH_SIZEの定数倍のデータ数に調整

        self.data.sort(key=get_binaryfile_number)

        if len(self.data) > config.MAX_LEN_TRAIN_DATA:
            self.data = self.data[:config.MAX_LEN_TRAIN_DATA]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (target)
                target      : tensor(1, R, C, H, W)

        """
        self.data_index = index

        target_path = self.data[index]

        target = self.load_tensor(target_path)
        anchor = target[0].unsqueeze(0)
        positive = target[1: 1 + config.NUM_POSITIVE]
        negative = target[1 + config.NUM_POSITIVE:]

        #TODO:rotmnist用の変数
        labels = None
        return anchor, positive, negative, labels

    def __len__(self):
        return len(self.data)

    def load_tensor(self, path):
        """
            Return:
                tensor: (Rotation, Channel, Height, Width)
        """
        tensor = torch.load(path)
        tensor = tensor.type(torch.float)
        return tensor

    @staticmethod
    def mean_context(context):
        return torch.mean(context, 0)

    @staticmethod
    def get_dummy_data(dummy_path):
        #TODO:should return tensor.zeros, more fast
        dummy = torch.load(dummy_path).type(torch.float)
        return dummy

    @staticmethod
    def is_date_change(path_list):
        """Check whether path_list have a specific date.
            Args:
                path_list (list): date_dataid.pt [20120101_0000.pt, ]
        """
        date_list = [x.split("_")[0] for x in path_list]
        if len(set(date_list)) == 1:
            return False
        else:
            return True

    def is_data_drop(self, path_list):
        """Check whether path_list have continuous dataid in time.
            Args:
                path_list (list): date_dataid.pt [20120101_0000.pt, ]
        """
        dataid_list = [int(x.split("_")[1].split(".pt")[0]) for x in path_list]

        if dataid_list != sorted(dataid_list):
            #IDが時間順に並んでいることが前提なので，これの確認
            raise ValueError("data is not sorted in time.")

        if sum(np.diff(dataid_list)) / (2*self.window) == 1:
            return False
        else:
            return True

    def check_sequential(self, path_list):
        """if sequential is True, check whether data is sequential.
        else, return True.

        Args:
            path_list ([type]): [description]

        Returns:
            [bool]: (sequential or not) or (not care about sequential)
        """
        if self.sequential:
            if self.is_date_change(path_list) or self.is_data_drop(path_list):
                return True
            return False
        else:
            return False
