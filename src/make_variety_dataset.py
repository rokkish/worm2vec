"""Make dataset which contains (original, rotated, negative) images.
Algorithm
    1. Load pickle as df
        df  | distance | original path | target path | rotation
        dic = {"original_date": original_date, "target_date": target_date, 
               "target_rotate": TARGET_ROTATE, "distance": distance}
    2. Load each path as tensor
    3. Concat tensors
    4. Save as pickle
        (N, C, H, W) = (original + rotated + negative, 1, 64, 64)
"""

import argparse
import glob
import torch
import pandas as pd
import get_logger
logger = get_logger.get_logger(name='make_variety_datasets', save_name="../log/logger/test_make_variety_datasets.log")

class DatasetMaker(object):
    def __init__(self, num_negative, load_K, num_rotate, save_path="varietydata", root_dir=""):
        self.num_negative = num_negative
        self.load_K = load_K
        self.num_rotate = num_rotate
        self.save_path = save_path
        self.root_dir = root_dir
        self.pkl_list = sorted(glob.glob("{}distance_table_compress_top{}/*".format(self.root_dir, self.load_K)))
        self.original_date_path = ""
        self.history_dir = self.root_dir + self.save_path
        self.history_pkl = sorted(glob.glob("{}*.pt".format(self.history_dir)))
        logger.debug("load: {}".format(len(self.pkl_list)))
        logger.debug("history: {}".format(len(self.history_pkl)))
    def make(self):
        skip_count = 0
        for i, pkl in enumerate(self.pkl_list):
            pkl_name = pkl.split("/")[-1].split(".")[0]

            if i % 2500 == 0:
                logger.debug("i:{}, per:{:.1f}, skip:{}".format(i, i/len(self.pkl_list)*100, skip_count))

            print("\r [process] {}/{}, skip:{}".format(i, len(self.pkl_list), skip_count), end="")

            if "{}{}.pt".format(self.history_dir, pkl_name) in self.history_pkl:
                skip_count += 1
                continue

            self.load_df(pkl)
            self.load_tensors()
            self.concat_tensors()
            self.save_as_pkl()

    def load_df(self, pkl):
        self.df = pd.read_pickle(pkl)
        #logger.debug(self.df.head())

    def load_tensors(self):
        self.tensors = []
        self.original_date_path = self.df["original_date"].iloc[0]

        for r in range(0, 36, 36//self.num_rotate):
            self.tensors.append(self.load_tensor(self.original_date_path, r))
            #logger.debug("path{}, rotate{}".format(self.original_date_path, r))

        for k in range(self.num_negative):
            target_date_path_i = self.df["target_date"].iloc[k]
            target_rotate_i = self.df["target_rotate"].iloc[k]
            #logger.debug("path{}, rotate{}".format(target_date_path_i, target_rotate_i))
            self.tensors.append(self.load_tensor(target_date_path_i, target_rotate_i))

    def load_tensor(self, path, rotate):
        """

        Args:
            path ([type]): [description]
            rotate ([type]): [description]

        Returns:
            [tensor]: pick one rotate from (R, C, H, W), return (1, C, H, W)
        """
        return torch.load(self.root_dir+"alldata/"+path+".pt")[rotate]

    def concat_tensors(self):
        """Concat tensors for tensor in tensors_list(=self.tensors)
        Returns:
            cat_tensors[tensor]: (Variety, C, H, W)
        """
        self.cat_tensors = torch.zeros(len(self.tensors), 1, self.tensors[0].shape[1], self.tensors[0].shape[2])
        for i, tensor_i in enumerate(self.tensors):
            self.cat_tensors[i] = tensor_i

    def save_as_pkl(self):
        """Save self.cat_tensors as pkl
        """
        torch.save(self.cat_tensors.byte(), self.root_dir + self.save_path+ "/" + self.original_date_path + ".pt")

def main(args):
    logger.info("start")
    import os
    os.makedirs(args.root_dir + args.save_path, exist_ok=True)
    maker = DatasetMaker(args.num_negative, load_K=args.load_K, num_rotate=args.num_rotate, save_path=args.save_path, root_dir=args.root_dir)
    maker.make()
    logger.info("end")

def chk(args):
    if args.num_rotate < 1 or args.num_rotate > 36:
        return False
    if args.num_negative < 1 or args.num_negative > 100:
        return False
    if args.load_K < args.num_negative:
        return False

    return True

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--load_K", type=int, default=5)
    parse.add_argument("--num_negative", type=int, default=4)
    parse.add_argument("--num_rotate", type=int, default=4)
    parse.add_argument("--save_path")
    parse.add_argument("--root_dir", default="../../data/processed/")
    args = parse.parse_args()
    if chk(args):
        main(args)
