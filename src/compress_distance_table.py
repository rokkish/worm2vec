"""Compress distance table made by get_distance_table.py
Algorithm
    1. Unzip distance table
    2. Load pickle
    3. Sort df in distance
    4. Pick top-K
    5  Delete unzip dir
    6. Save df as pkl
"""

import os
import glob
import pandas as pd
import numpy as np
import shutil
import zipfile
import time
import argparse
from multiprocessing import Pool
import get_logger
logger = get_logger.get_logger(name='compress_distance_table', save_name="../log/logger/test_compress.log")


def pdreadpkl(pkl_path):
    return pd.read_pickle(pkl_path)
def read_pkl_map(file_list):
    df = pd.concat(map(pdreadpkl, file_list))
    return df
def read_pkl_map_multi(file_list, div):
    p = Pool(os.cpu_count()//div)
    df = pd.concat(p.map(pdreadpkl, file_list))
    #TODO:使ったら，pool終了させる
    return df

class Compressor(object):
    def __init__(self, K):
        self.K = K
        self.zip_list = sorted(glob.glob("../../data/processed/distance_table/*"))
        self.unzip_dir = "../../data/processed/distance_table/temp/"
        self.history_dir = "../../data/processed/distance_table_compress_top{}/".format(self.K)
        self.history_pkl = sorted(glob.glob(self.history_dir + "*"))
        logger.debug("zip list:{}".format(len(self.zip_list)))

    def compress_all(self):
        for i, zip_i in enumerate(self.zip_list):
            zip_name = zip_i.split("/")[-1].split(".")[0]

            #TODO:skip
            if "{}{}.pkl".format(self.history_dir, zip_name) in self.history_pkl:
                continue

            self.compress(zip_i)

            if i % 100 == 0:
                logger.debug("i:{}, per:{:3f}".format(i, i/len(self.zip_list)*100))

            self.save_df_as_pickle(zip_name)

    def compress(self, zip_i):
        self.mkdir()
        #logger.debug("end mkdir")
        self.unzip(zip_i)
        #logger.debug("end unzip")
        self.load_pickle()
        #logger.debug("end load_pickle")
        self.concat_df()
        #logger.debug("end concat_df")
        self.sort()
        #logger.debug("end sort")
        self.pick_topk()
        #logger.debug("end pick_topk")
        self.delete_unzip_dir()
        #logger.debug("end delete_unzip_dir")

    def mkdir(self):
        os.makedirs(self.unzip_dir, exist_ok=True)

    def unzip(self, zip_i):
        with zipfile.ZipFile(zip_i) as existing_zip:
            existing_zip.extractall(self.unzip_dir)

    def load_pickle(self):
        self.ls_pkl = glob.glob(self.unzip_dir + "*.pkl")

    def concat_df(self):
        #self.df = read_pkl_map_multi(self.ls_pkl, div=32)
        self.df = read_pkl_map(self.ls_pkl)

    def sort(self):
        self.df = self.df.sort_values("distance")
        self.df_reset_index()
    def pick_topk(self):
        self.df = self.df.iloc[-self.K:]
        self.df_reset_index()
    def delete_unzip_dir(self):
        shutil.rmtree(self.unzip_dir)
    def save_df_as_pickle(self, zip_name):
        save_name_pkl = "../../data/processed/distance_table_compress_top{}/{}.pkl".format(self.K, zip_name)
        self.df.to_pickle(save_name_pkl)
    def df_reset_index(self):
        self.df = self.df.reset_index(drop=True)

def main(args):
    logger.info("start")
    os.makedirs("../../data/processed/distance_table_compress_top{}".format(args.K), exist_ok=True)
    comp = Compressor(args.K)
    comp.compress_all()
    logger.info("end")


def check(args):
    if args.K > 100 or args.K < 1:
        print("out of bound.0<K<101")
        return False
    return True


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-K", type=int, default=5)
    args = parse.parse_args()
    if check(args):
        main(args)