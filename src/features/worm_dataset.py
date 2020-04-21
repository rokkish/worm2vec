"""class dataset"""

import glob
import torch
import torch.utils.data
import config
import get_logger
import numpy as np
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
        self.data_index = 0

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
        self.data_index = index
        if index - self.window < 0 or index + 1 + self.window > len(self.data):
            dummy_path = self.data[index]
            dummy = torch.load(dummy_path).type(torch.float)
            return {config.error_idx: dummy}

        target_path = self.data[index]
        left_context_path = self.data[index - self.window:index]
        right_context_path = self.data[index + 1:index + 1 + self.window]

        target = torch.load(target_path)
        target = target.type(torch.float)
        target = target.unsqueeze(0)
        context = self.load_tensor(left_context_path + right_context_path)
        context = context.type(torch.float)
        context = self.mean_context(context)
        context = context.unsqueeze(0)

        return {self.data_index: torch.cat([target, context], dim=0)}

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

    @staticmethod
    def mean_context(context):
        return torch.mean(context, 0)

    @staticmethod
    def get_dummy_data(dummy_path):
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
