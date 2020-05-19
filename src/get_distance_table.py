"""
    Get distance table preprocessed data
        1. Load tensor (N:300000, Rotation:36, Channel:1, Height:64, Width:64)

        2. Sample 2 images (2, 36, 1, 64, 64)

        3. Get distance between A and A'
            1.. Transform A(36, 1, 64, 64) into A'(36, 36, 1, 64, 64)
            2.. dist[i][theta] = MSE(A'[i] - A) for i in range(36)
            3.. df = {id_A:id(A[theta]), id_A':id(A'[i]), 
                      dist:dist[i][theta]} for i for theta

        4. Save as dataframe
            | index |  id_A(ymdhm_id_theta) |  id_A'(ymdhm_id_theta) | dist |
                0      20120101_000000_00       20120101_000001_00     0.331
"""

import time
import glob
import torch
import torch.nn as nn
import pandas as pd
import argparse

import config
import features.sort_index as sort_index
from trainer import Trainer
import get_logger
logger = get_logger.get_logger(name='get_distance_table')
DUMMY = torch.zeros(36, 36, 1, config.IMG_SIZE, config.IMG_SIZE)


class WormDataset_get_table(torch.utils.data.Dataset):
    """
        clone from "class WormDataset"
        diff:
            Load all of raw data to preprocess.
    """

    def __init__(self, root, transform=None, START_ID=0, END_ID=1):

        self.root = root
        self.transform = transform
        self.START_ID = START_ID
        self.END_ID = END_ID

        self.alldata = glob.glob(self.root+"/*.pt")
        self.alldata.sort(key=sort_index.get_binaryfile_number)
        self.data = self.alldata[self.START_ID:self.END_ID]

        logger.debug("head of self.data %s" % (self.data[:2]))
        logger.debug("tail of self.data %s" % (self.data[-2:]))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image)
        """
        img_path = self.data[index]
        #['..', '..', 'data', 'processed', 'alldata', '201302081603_000000.pt']
        img_date_id = img_path.split("/")[5].split(".pt")[0]

        try:
            img = torch.load(img_path)
            copy_img = self.copy_cat(img)
            #cat_img = self.cat(img, copy_img)

        except IOError:
            logger.debug("IOError")
            return {config.error_idx: DUMMY}

        return {img_date_id: copy_img.type(torch.float)}

    def __len__(self):
        return len(self.data)

    def get_allpath(self):
        return self.alldata

    @staticmethod
    def copy_cat(tensor):
        """Copy tensor.
            Args:
                tensor(36, 1, 64, 64)
            Return:
                tensor(36, 36, 1, 64, 64)
        """
        for i in range(tensor.shape[0]):
            tensori = tensor[i].unsqueeze(dim=0)
            tensori2 = torch.cat([tensori, tensori], dim=0)
            tensori4 = torch.cat([tensori2, tensori2], dim=0)
            tensori8 = torch.cat([tensori4, tensori4], dim=0)
            tensori16 = torch.cat([tensori8, tensori8], dim=0)
            tensori32 = torch.cat([tensori16, tensori16], dim=0)
            tensori36 = torch.cat([tensori32, tensori4], dim=0)
            tensori36 = tensori36.unsqueeze(dim=0)
            if i == 0:
                cat_tensor = tensori36
            else:
                cat_tensor = torch.cat([cat_tensor, tensori36], dim=0)
        return cat_tensor

    @staticmethod
    def cat(original, copy):
        """
            Args
                original: (36, 1, 64, 64)
                copy    : (36, 36, 1, 64, 64)
        """
        original = original.unsqueeze(dim=0)
        tensor = torch.cat([original, copy], dim=0)
        return tensor


class Get_distance_table(object):
    
    def __init__(self, process_id, save_name):
        self.process_id = process_id
        self.save_name = save_name
        self.START_ID, self.END_ID = self.count_img()
        self.device = torch.device("cuda:" + str(self.process_id) if torch.cuda.is_available() else "cpu")

    def count_img(self):
        """Count num dataset, and return (start, end) id to divide data. 
            Arg:
                process_id (int): No.[0, 1, 2, 3] of docker container.
        """

        img_list = glob.glob("../../data/processed/alldata/*")

        START_ID = len(img_list) // 4 * self.process_id
        END_ID = len(img_list) // 4 * (self.process_id + 1)

        return START_ID, END_ID

    def load_datasets(self):
        """ Set dataset """
        dataset = WormDataset_get_table(root="../../data/processed/alldata", transform=None, START_ID=self.START_ID, END_ID=self.END_ID)

        allpath = dataset.get_allpath()

        """ Dataloader """
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False)

        return loader, allpath

    def calc_distance(self, loader, allpath):
        """Preprocess datasets
            Arg:
                loader(datasets):iterator of data
                allpath(list)   :[path, ..., path]
        """
        init_t = time.time()

        count_skip_img = 0

        for data_i, data_dic in enumerate(loader):

            logger.debug("allpath:{}".format(len(allpath)))
            allpath = allpath[data_i + self.START_ID:]
            logger.debug("data_i:{}".format(data_i + self.START_ID))

            date, data = Trainer.get_data_from_dic(data_dic)
            #data = data.to(self.device)

            if (data_i + self.START_ID) % 1000 == 0:
                logger.debug("[%d] %d/%d \t Load&Save Processd :%d sec" %
                            (process_id, data_i + self.START_ID, self.END_ID, time.time() - init_t))

            if date == config.error_idx:
                logger.debug("Skip this batch beacuse window can't load data")
                logger.debug("Skip Data:%s, Date:%s" % (data.shape, date))
                count_skip_img += 1
                continue

            mse = self.get_mse_epoch(data, allpath)
            logger.debug("filename :{}".format(date))
            logger.debug("datashape:{}".format(data.shape))

            break

        logger.debug("[%d] %d/%d \t Finish Processd :%d sec" %
                    (self.process_id, data_i + self.START_ID, self.END_ID, time.time() - init_t))

    def get_mse_epoch(self, x, allpath):
        """Get mse
            Args:
                x.shape = (1, 36, 36, 1, 64, 64)
                len(allpath) = DataLength:300000
        """
        for i in range(len(allpath)):
            target_y = torch.load(allpath[i]).type(torch.float)
            for batch in range(x.shape[2]):
                input_x = x[0, batch]
                mse = self.get_mse_batch(input_x, target_y)
            break

        return mse

    def get_mse_batch(self, x, y):
        """Get mse
            Args:
                x.shape = [36, 1, 64, 64]
                y.shape = [36, 1, 64, 64]
            Return:
                mse.shape = [36]
        """
        #x, y = x.to(self.device), y.to(self.device)
        loss = nn.MSELoss(reduction="none")
        mse = loss(x, y)
        # mse:(36, 1, 64, 64)
        mse = mse.squeeze()
        mse = torch.sum(torch.sum(mse, dim=1), dim=1)
        return mse

    def save_as_df(self, dic):
        """
            Args:
                dic={
                    original_date_i:[],
                    target_date_i:[],
                    dist:[],
                }
        """
        start, end = min(dic), max(dic)
        df = pd.DataFrame(dic)
        df.to_csv("../../data/processed/dist_from{}to{}.csv".format(start, end))


def main(args):
    """Load datasets, Do preprocess()
    """
    gettabler = Get_distance_table(args.process_id, args.save_name)
    loader, allpath = gettabler.load_datasets()
    gettabler.calc_distance(loader, allpath)


def chk(args):
    if args.process_id > 3 or args.process_id < 0:
        raise ValueError("input [0 ~ 3] process_id")
        return False
    return True


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--process_id", type=int, default=0,
                       help="input 0~3 for pararell docker container")
    parse.add_argument("--save_name", default="test")

    args = parse.parse_args()

    if chk(args):
        main(args)
