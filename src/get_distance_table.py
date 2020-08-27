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
import random

import config
import features.sort_index as sort_index
from trainer import Trainer
from visualization.save_images_gray_grid import save_images_grid
import get_logger
logger = get_logger.get_logger(name='get_distance_table', save_name="../log/logger/get_distance_table.log")
DUMMY = torch.zeros(36, 1, config.IMG_SIZE, config.IMG_SIZE)
TARGET_ROTATE = list(range(0, 36))


def zip_dir():

    import shutil
    root_dirs = glob.glob("../../data/processed/distance_table/*")

    for i, root_dir_i in enumerate(root_dirs):
        if ".zip" in root_dir_i or root_dir_i + ".zip" in root_dirs:
            continue

        shutil.make_archive(root_dir_i, "zip", root_dir=root_dir_i)
        #logger.debug("rm {}".format(root_dir_i))
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
        self.history_zip = glob.glob("../../data/processed/distance_table/*")

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

        if self.is_already_calculated(img_date_id):
            return {config.error_idx: DUMMY}

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
                      (Rotation, Channel, H, W)
            Return:
                tensor(36, 1, 64, 64)
                      (Sameimg, Channel, H, W)
        """
        tensors = torch.zeros(36, 1, config.IMG_SIZE, config.IMG_SIZE)
        for i in range(tensors.shape[0]):
            tensors[i] = tensor[0]
        return tensors

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

    #@staticmethod
    def is_already_calculated(self, date):
        """chk whether date in ~/distance_table/*.zip
            Args
                date = "201201021359_000000"
            Return
        """
        if "../../data/processed/distance_table/" + date + ".zip" in self.history_zip:
            return True
        return False

class Get_distance_table(object):
    
    def __init__(self, process_id, gpu_id, save_name, max_num_of_original_data, max_num_of_pair_data):
        self.process_id = process_id
        self.gpu_id = gpu_id
        self.save_name = save_name
        self.MAX_NUM_OF_ORIGINAL_DATA = max_num_of_original_data
        self.MAX_NUM_OF_PAIR_DATA = max_num_of_pair_data
        self.START_ID, self.END_ID = self.count_img()
        self.device = torch.device("cuda:" + str(self.gpu_id) if torch.cuda.is_available() else "cpu")
        self.max_distance_list = []#original一枚に対する全ペア間の最大距離

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
        skip_count = 0
        for data_i, data_dic in enumerate(loader):

            #logger.debug("allpath:{}, data_i:{}".format(len(allpath), data_i + self.START_ID))

            date, data = Trainer.get_data_from_dic(data_dic)

            print("\r [process] {}/{}".format(data_i + self.START_ID, self.END_ID), end="")

            if data_i % 1000 == 0:
                logger.debug("[{}] {}/{} \t Processd :{:.3f} sec, skip:{}".format
                            (self.process_id, data_i + self.START_ID, self.END_ID, time.time() - init_t, skip_count))

            if date == config.error_idx:
                skip_count += 1
                #logger.debug("Skip this batch beacuse window can't load data")
                #logger.debug("Skip Data:%s, Date:%s" % (data.shape, date))
                continue

            #if self.is_already_calculated(date):
            #    continue

            if data_i >= self.MAX_NUM_OF_ORIGINAL_DATA:
                break

            random_path_list = self.get_random_paths(np.array(allpath), self.MAX_NUM_OF_PAIR_DATA, data_i)
            #logger.debug("allpath:{}, data_i:{}".format(len(random_path_list), data_i))

            self.get_mse_epoch(data, date, random_path_list)
            #logger.debug("GPU {}, max dist:{}".format(self.process_id, max(self.max_distance_list)))
            self.max_distance_list = []
            zip_dir()   #TODO:並列実行すると，予期せぬ挙動になる．

        logger.debug("GPU [%d] %d/%d \t Finish Processd :%f sec" %
                    (self.process_id, data_i + self.START_ID, self.END_ID, time.time() - init_t))

    def get_mse_epoch(self, x, x_date, allpath):
        """Get mse

        Args:
            x (tensor): Input tensor (1, 36, 1, 64, 64)
            x_date (str): To save as file name like "201201021359_000000"
            allpath (list(str)): file paths of target tensor. Length is 300000 - calc_distance.data_i.

        Returns:
            None
        """
        input_x = x[0].to(self.device)

        values = [(i, pathi) for i, pathi in enumerate(allpath)]

        #logger.debug("load tensors")
        tensors_y = self.load_target_tensor(allpath)
        tensors_y = tensors_y.to(self.device)
        #logger.debug("loaded tensors:{}".format(tensors_y.shape))

        def get_mse_batch_map(values, tensors_y=tensors_y):
            idx, pathi = values[0], values[1]
            target_y = tensors_y[idx]

            #if idx % (self.MAX_NUM_OF_PAIR_DATA//10) == 0:
            #logger.debug("GPU[{}] get_mse_batch_map idx:{}".format(self.process_id, idx))

            y_date = get_date_to_split_path(pathi)

            mse = self.get_mse_batch(input_x, target_y)

            dic = self.mk_dictionary_to_save(x_date, y_date, mse)

            return dic, idx

        map_object = map(get_mse_batch_map, values)
        [self.save_as_df(dic, original_date=x_date) for dic, idx in map_object]

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

    def load_target_tensor(self, allpath):
        """load tensor, concat.

        Args:
            allpath (list[str]): path of tensor

        Returns:
            newTensor (tensor): (N, 36, 1, 64, 64)
        """
        t = torch.load(allpath[0])
        if len(t.shape) != 4:
            raise ValueError("not match shape")

        newTensor = torch.zeros(len(allpath), t.shape[0], t.shape[1], t.shape[2], t.shape[3])

        for idx, pathi in enumerate(allpath):

            #if idx % (self.MAX_NUM_OF_PAIR_DATA//10) == 0:
            #logger.debug("GPU[{}] load_target_tensor_map idx:{}".format(self.process_id, idx))

            newTensor[idx] = torch.load(pathi).unsqueeze(0)

        return newTensor.type(torch.float)

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
                mse     : list [36]
        """
        original_date = []
        target_date = []
        distance = []

        original_date.extend([x_date]*36)
        target_date.extend([y_date]*36)
        distance.extend(np.array(mse.cpu()).astype("int32"))

        dic = {"original_date": original_date, "target_date": target_date, 
               "target_rotate": TARGET_ROTATE, "distance": distance}
        #logger.debug("Len of original:{}, target:{}, mse:{}".format(len(original_date), len(targe_date), len(distance)))
        return dic

    def save_as_df(self, dic, original_date):
        """
            Args:
                dic={
                    original_date:[],
                    target_date:[],
                    target_rotate:[],
                    distance:[],
                }
        """
        target_date = dic["target_date"][0]
        df = pd.DataFrame(dic)
        self.max_distance_list.append(df["distance"].max())
        os.makedirs("../../data/processed/distance_table/{}".format(original_date), exist_ok=True)
        df.to_pickle("../../data/processed/distance_table/{}/dist_from_{}.pkl".format(original_date, target_date))

    def save_as_img(self, input_x, target_y):
        """Save data as img
            Args
                input_x.shape  = [36, 1, 64, 64]
                target_y.shape = [36, 1, 64, 64]
        """
        save_images_grid(input_x.cpu(), nrow=config.nrow, scale_each=True, global_step=0, tag_img="test/input_x", writer=None, filename="../results/input.png")
        save_images_grid(target_y.cpu(), nrow=config.nrow, scale_each=True, global_step=0, tag_img="test/target_y", writer=None, filename="../results/target.png")

    @staticmethod
    def is_already_calculated(date):
        """chk whether date in ~/distance_table/*.zip
            Args
                date = "201201021359_000000"
            Return
        """
        dirs = glob.glob("../../data/processed/distance_table/*")
        if "../../data/processed/distance_table/" + date + ".zip" in dirs:
            #logger.debug("Skip:{}".format(date))
            return True
        return False

    @staticmethod
    def get_random_paths(allpath, N, start_index):
        #original_imgを対象外とするため
        start_index = start_index % 10 + 1
        random_indexs = random.sample(range(start_index, len(allpath), 10), k=N)
        return allpath[random_indexs]

def main(args):
    """Load datasets, Do preprocess()
    """
    logger.info("start")
    gettabler = Get_distance_table(args.process_id, args.gpu_id, args.save_name, args.max_original, args.max_pair)
    loader, allpath = gettabler.load_datasets()
    gettabler.calc_distance(loader, allpath)
    #zip_dir()
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
                       help="input 0~3 for pararell docker container to select splited data")
    parse.add_argument("--gpu_id", default="0")
    parse.add_argument("--save_name", default="test")
    parse.add_argument("--max_original", type=int, default=1)
    parse.add_argument("--max_pair", type=int, default=10000)

    args = parse.parse_args()

    if chk(args):
        main(args)
