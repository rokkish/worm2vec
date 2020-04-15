"""
    Preprocess raw data
        : Tobinary, Fill hole, Labelling, Padding
"""

import os
import time
import glob
from PIL import Image
import torch
from torchvision import transforms
import argparse

import config
from features.worm_transform import ToBinary, FillHole, Labelling, Rotation, Padding, Resize, ToNDarray
import features.sort_index as sort_index
from trainer import Trainer
import get_logger
logger = get_logger.get_logger(name='preprocess')


class WormDataset_prepro(torch.utils.data.Dataset):
    """
        clone from "class WormDataset"
        diff:
            Load all of raw data to preprocess.
    """

    def __init__(self, root, transform=None, START_ID=0, END_ID=1):

        self.root = root
        self.transform = transform
        self.data = []
        self.START_ID = START_ID
        self.END_ID = END_ID

        data_dirs_all = glob.glob(self.root + "/*")
        data_dirs = data_dirs_all

        for dir_i in data_dirs:
            img_path_ls = glob.glob(dir_i+"/main/*.bmp")
            img_path_ls.sort(key=sort_index.get_file_number)
            self.data.extend(img_path_ls)
        self.data = self.data[self.START_ID:self.END_ID]
        logger.debug("head of self.data %s" % (self.data[:2]))
        logger.debug("tail of self.data %s" % (self.data[-2:]))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image)
        """
        img = self.data[index]
        #FIXME:not smart['..', '..', 'data', 'Tanimoto_eLife_Fig3B', '201302081603', 'main', 'img3317.bmp']
        img_date = img.split("/")[4]

        try:
            img = Image.open(img)
        except IOError:
            print("IOError")
            dummy = torch.zeros(36, 1, config.IMG_SIZE, config.IMG_SIZE)
            return {config.error_idx: dummy}

        if self.transform is not None:
            img = self.transform(img)

        return {img_date: img}

    def __len__(self):
        return len(self.data)


def load_datasets(START_ID, END_ID):
    """ Set transform """
    worm_transforms = transforms.Compose([
        ToBinary(),
        FillHole(),
        Labelling(),
        Padding(),
        Rotation(),
        Resize((config.IMG_SIZE, config.IMG_SIZE))])
        #transforms.ToTensor()])

    """ Set dataset """
    dataset = WormDataset_prepro(root="../../data/Tanimoto_eLife_Fig3B",
                                 transform=worm_transforms,
                                 START_ID=START_ID,
                                 END_ID=END_ID)

    """ Dataloader """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False)

    return loader


def count_img(process_id):
    """Count num dataset, and return (start, end) id to divide data. 
        Arg:
            process_id (int): No.[0, 1, 2, 3] of docker container.
    """
    img_list = []

    data_dirs = glob.glob("../../data/Tanimoto_eLife_Fig3B/*")

    for dir_i in data_dirs:
        img_list.extend(glob.glob(dir_i + "/main/*"))

    START_ID, END_ID = len(img_list) // 4 * process_id, \
        len(img_list) // 4 * (process_id + 1)

    logger.debug("[%d] load data from %d to %d all:%d" %
                 (process_id, START_ID, END_ID, len(img_list)))

    return START_ID, END_ID


def preprocess(START_ID, END_ID, loader, process_id, save_name):
    """Preprocess datasets
        Arg:
            START_ID, END_ID (int)          :(S, E) range of preprocessing datasets
            loader           (datasets)     :iterator of data
            process_id       (int)          :0, 1, 2, 3
            save_name        (str)          :dir name
    """
    init_t = time.time()

    count_delete_img = 0

    for data_i, data_dic in enumerate(loader):

        date, data = Trainer.get_data_from_dic(data_dic)

        if (data_i + START_ID) % 1000 == 0:
            logger.debug("[%d] %d/%d \t Load&Save Processd :%d sec" %
                         (process_id, data_i + START_ID, END_ID, time.time() - init_t))

        if date == config.error_idx:
            logger.debug("Skip this batch beacuse window can't load data")
            logger.debug("Skip Data:%s, Date:%s" % (data.shape, date))
            count_delete_img += 1
            continue

        data = data[0]
        dir_i = "../../data/" + save_name + "/" + date
        tensor_name = "{}/tensor_{:0=6}.pt".format(dir_i, data_i + START_ID)

        if os.path.isfile(tensor_name):
            continue

        if len(data.shape) != 4:
            count_delete_img += 1

        elif torch.sum(data) == 0:
            count_delete_img += 1

        else:
            os.makedirs(dir_i, exist_ok=True)
            torch.save(data, tensor_name)

    logger.debug("[%d] delete %d img" % (process_id, count_delete_img))
    logger.debug("[%d] %d/%d \t Finish Processd :%d sec" %
                 (process_id, data_i + START_ID, END_ID, time.time() - init_t))


def main(args):
    """Load datasets, Do preprocess()
    """
    START_ID, END_ID = count_img(args.process_id)

    loader = load_datasets(START_ID, END_ID)

    preprocess(START_ID, END_ID, loader, args.process_id, args.save_name)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--process_id", type=int, default=0,
                       help="input 0~3 for pararell docker container")
    parse.add_argument("--save_name", default="processed")

    args = parse.parse_args()

    main(args)
