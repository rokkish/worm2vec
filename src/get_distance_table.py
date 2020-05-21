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
import os
import time
import glob
import torch
from torchvision import transforms #Need to escape error AttributeError: module 'torch.utils' has no attribute 'data'
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
from tensorboardX import SummaryWriter

import config
import features.sort_index as sort_index
from trainer import Trainer
from visualization.save_images_gray_grid import save_images_grid
import get_logger
logger = get_logger.get_logger(name='get_distance_table')
DUMMY = torch.zeros(36, 36, 1, config.IMG_SIZE, config.IMG_SIZE)


def zip_dir():

    import shutil
    root_dirs = glob.glob("../../data/processed/distance_table/*")

    for i, root_dir_i in enumerate(root_dirs):
        if ".zip" in root_dir_i or root_dir_i + ".zip" in root_dirs:
            continue

        shutil.make_archive(root_dir_i, "zip", root_dir=root_dir_i)
        logger.debug("rm {}".format(root_dir_i))
        shutil.rmtree(root_dir_i)


def get_date_to_split_path(path):
    #['..', '..', 'data', 'processed', 'alldata', '201302081603_000000.pt']
    return path.split("/")[5].split(".pt")[0]


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
        img_date_id = get_date_to_split_path(img_path)

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
    
    def __init__(self, process_id, save_name, max_num_of_original_data, max_num_of_pair_data):
        self.process_id = process_id
        self.save_name = save_name
        self.MAX_NUM_OF_ORIGINAL_DATA = max_num_of_original_data
        self.MAX_NUM_OF_PAIR_DATA = max_num_of_pair_data
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

        allpath = allpath[self.START_ID + 1:]
        count_skip_img = 0

        for data_i, data_dic in enumerate(loader):

            allpath_fromi = allpath[data_i:]
            logger.debug("allpath:{}".format(len(allpath_fromi)))
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

            self.get_mse_epoch(data, date, allpath_fromi)
            logger.debug("filename :{}".format(date))
            logger.debug("data:{}".format(data.shape))

            if data_i >= self.MAX_NUM_OF_ORIGINAL_DATA:
                break

        logger.debug("GPU Used")
        logger.debug("[%d] %d/%d \t Finish Processd :%f sec" %
                    (self.process_id, data_i + self.START_ID, self.END_ID, time.time() - init_t))

    def get_mse_epoch(self, x, x_date, allpath):
        """Get mse
            Args:
                x.shape = (1, 36, 36, 1, 64, 64)
                x_date = "201201021359_000000"
                len(allpath) = DataLength:300000 - calc_distance.data_i
        """
        for i, pathi in enumerate(allpath):

            target_y = torch.load(pathi).type(torch.float)
            y_date = get_date_to_split_path(pathi)
            mse = [None]*36
            x, target_y = x.to(self.device), target_y.to(self.device)

            for batch in range(x.shape[2]):

                input_x = x[0, batch]
                mse[batch] = self.get_mse_batch(input_x, target_y)

            dic = self.mk_dictionary_to_save(x_date, y_date, mse)
            self.save_as_df(dic, dir_name=x_date)

            if i % (self.MAX_NUM_OF_PAIR_DATA//10) == 0:
                logger.debug("i:{}, dic0:{}".format(i, dic["target_date"][0]))

            if i > self.MAX_NUM_OF_PAIR_DATA:
                break

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

    @staticmethod
    def add_rotation_name_to_date(date):
        """add rotation info to date, return date_list[0, ..., 35]
            Args
                date = "201201021359_000000"
            Return
                list = ["201201021359_000000_00", ..., "201201021359_000000_35"]
        """
        ls = []
        for i in range(36):
            ls.append(date + "_{:0=2}".format(i))
        return ls

    def mk_dictionary_to_save(self, x_date, y_date, mse):
        """
            Args
                x_date  : str
                y_date  : str
                mse     : list(list) [36, 36]
        """
        x_ls = self.add_rotation_name_to_date(x_date)
        y_ls = self.add_rotation_name_to_date(y_date)
        original_date = []
        targe_date = []
        distance = []
        for i in range(36):
            original_date.extend([x_ls[i]]*36)
            targe_date.extend(y_ls)
            distance.extend(np.array(mse[i].cpu()))
        dic = {"original_date": original_date, "target_date": targe_date, "distance": distance}
        #logger.debug("Len of original:{}, target:{}, mse:{}".format(len(original_date), len(targe_date), len(distance)))
        return dic

    def save_as_df(self, dic, dir_name):
        """
            Args:
                dic={
                    original_date:[],
                    target_date:[],
                    distance:[],
                }
        """
        start, end = dic["target_date"][0], dic["target_date"][-1]
        df = pd.DataFrame(dic)
        os.makedirs("../../data/processed/distance_table/{}".format(dir_name), exist_ok=True)
        df.to_pickle("../../data/processed/distance_table/{}/dist_from{}to{}.pkl".format(dir_name, start, end))

    def save_as_img(self, input_x, target_y):
        """Save data as img
            Args
                input_x.shape  = [36, 1, 64, 64]
                target_y.shape = [36, 1, 64, 64]
        """
        save_images_grid(input_x.cpu(), nrow=config.nrow, scale_each=True, global_step=0, tag_img="test/input_x", writer=None, filename="../results/input.png")
        save_images_grid(target_y.cpu(), nrow=config.nrow, scale_each=True, global_step=0, tag_img="test/target_y", writer=None, filename="../results/target.png")


def main(args):
    """Load datasets, Do preprocess()
    """
    logger.info("start")
    gettabler = Get_distance_table(args.process_id, args.save_name, args.max_original, args.max_pair)
    loader, allpath = gettabler.load_datasets()
    gettabler.calc_distance(loader, allpath)
    zip_dir()
    logger.info("end")


def chk(args):
    if args.process_id > 3 or args.process_id < 0:
        raise ValueError("input [0 ~ 3] process_id")
    if args.max_original < 0:
        raise ValueError("max_original under 1")
    return True


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--process_id", type=int, default=0,
                       help="input 0~3 for pararell docker container")
    parse.add_argument("--save_name", default="test")
    parse.add_argument("--max_original", type=int, default=1)
    parse.add_argument("--max_pair", type=int, default=10000)

    args = parse.parse_args()

    if chk(args):
        main(args)
