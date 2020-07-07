"""class dataset"""

import glob
import torch
import torch.utils.data
import config
import get_logger
import numpy as np
logger = get_logger.get_logger(name='rot_mnist_dataset')

class RotMnistDataset(torch.utils.data.Dataset):
    """
        Def Dataset
    """

    def __init__(
            self, 
            root, 
            train=True, 
            num_pos=config.NUM_POSITIVE, 
            num_neg=config.NUM_NEGATIVE,
            batch_size=config.BATCH_SIZE,
            max_len=config.MAX_LEN_TRAIN_DATA,
            transform=None,
            img_size=config.IMG_SIZE
            ):

        """[summary]

        Args:
            root ([type]): [description]
            train (bool, optional): [description]. Defaults to True.
            transform ([type], optional): [description]. Defaults to None.

        Varibale:
            self.data : (N, H*W)
            self.labels : (N, class)

        """
        self.root = root    # root_dir \Tanimoto_eLife_Fig3B or \unpublished control
        self.train = train  # training set or test set
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.batch_size = batch_size
        self.max_len = max_len
        self.transform = transform
        self.img_size = img_size

        self.data = []
        self.data_index = 0
        self.count_skip_data = 0

        if self.train:
            self.load_data = np.loadtxt(self.root + "mnist_all_rotation_normalized_float_train_valid.amat")
            self.data, self.labels = self.load_data[:, :-1], self.load_data[:, -1]

        else:
            self.load_data = np.loadtxt(self.root + "mnist_all_rotation_normalized_float_test.amat")
            self.data, self.labels = self.load_data[:, :-1], self.load_data[:, -1]

            # BATCH_SIZEの定数倍のデータ数に調整(test dataのみ)
            if len(self.data) % self.batch_size == 0:
                self.data = self.data[:self.max_len]
            else:
                self.data = self.data[:-(len(self.data) % self.batch_size)]

        if len(self.data) > self.max_len:
            self.data = self.data[:self.max_len]

        self.labels = self.labels[:len(self.data)]

        #logger.debug("len of data:{}, {}".format(len(self.data), len(self.labels)))

    def __getitem__(self, index):
        """[summary]

        Args:
            index ([int]): [description]

        Returns:
            anchor: (H*W) -> (1, 1, H, W)
            positive: (H*W) -> (n, 1, H, W)
            negative: (H*W) -> (p, 1, H, W)
        """
        self.data_index = index

        anchor, anchor_label = self.data[index], self.labels[index]
        pos_idxs, neg_idxs = self.random_choice(anchor_label, index)

        positive = np.zeros((self.num_pos, 28, 28))
        negative = np.zeros((self.num_neg, 28, 28))

        for i, p in enumerate(pos_idxs):
            data_i = self.data[p]
            positive[i, :, :] = np.reshape(data_i, (28, 28)).T
        for i, n in enumerate(neg_idxs):
            data_i = self.data[n]
            negative[i, :, :] = np.reshape(data_i, (28, 28)).T

        # reshape, new axis
        anchor = np.reshape(anchor, (28, 28)).T
        anchor = np.reshape(anchor, (1, 28, 28))

        if self.transform is not None:
            anchor, positive, negative = self.transform(anchor), self.transform(positive), self.transform(negative)

        # reshape, new axis
        anchor = np.reshape(anchor, (1, 1, self.img_size, self.img_size))
        positive = np.reshape(positive, (positive.shape[0], 1, self.img_size, self.img_size))
        negative = np.reshape(negative, (negative.shape[0], 1, self.img_size, self.img_size))
        # totensor
        anchor = torch.from_numpy(anchor).type(torch.float)
        positive = torch.from_numpy(positive).type(torch.float)
        negative = torch.from_numpy(negative).type(torch.float)

        labels_batch = self.get_labels_batch(index, pos_idxs, neg_idxs)
        return anchor, positive, negative, labels_batch

    def __len__(self):
        return len(self.data)

    def random_choice(self, anchor_label, anchor_idx):
        """ランダムサンプリングしたpos, negのidxを返す

        Args:
            anchor_label ([int]): [description]
            anchor_idx ([int]): [description]

        Returns:
            pos_idxs [list]: [description]
            neg_idxs [list]: [description]
        """
        import random
        pos_idxs, neg_idxs = [], []
        count_p = 0
        count_n = 0
        rand_idx_list = [random.randint(0, self.data.shape[0] - 1) for i in range(1000)]

        for i in rand_idx_list:

            if i == anchor_idx:
                continue

            if self.labels[i] == anchor_label and count_p < self.num_pos: #positive
                pos_idxs.append(i)
                count_p += 1

            if self.labels[i] != anchor_label and count_n < self.num_neg: #negative
                neg_idxs.append(i)
                count_n += 1

            if count_p >= self.num_pos and count_n >= self.num_neg:
                break

        return pos_idxs, neg_idxs

    def get_labels_batch(self, anc_idx, pos_idxs, neg_idxs):
        return [str(self.labels[idx]) for idx in [anc_idx] + pos_idxs + neg_idxs]

