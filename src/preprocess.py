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

class WormDataset_prepro(torch.utils.data.Dataset):
    """
        clone from "class WormDataset"
        diff:
            Load all of raw data to preprocess.
    """

    def __init__(self, root, transform=None, START_ID=0, END_ID=1):

        self.root = root    # root_dir \Tanimoto_eLife_Fig3B or \unpublished control
        self.transform = transform
        self.data = []
        self.START_ID = START_ID
        self.END_ID = END_ID

        data_dirs_all = glob.glob(self.root + "/*")
        data_dirs = data_dirs_all

        for dir_i in data_dirs:
            self.data.extend(glob.glob(dir_i + "/main/*"))
        self.data = self.data[self.START_ID:self.END_ID]

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
            return 0

        if self.transform is not None:
            img = self.transform(img)

        return img, img_date

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
        transform=worm_transforms, START_ID=START_ID, END_ID=END_ID)

    """ Dataloader """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False)

    return loader

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--process_id", type=int, default=0,
                help="input 0~3 for pararell docker container")
    parse.add_argument("--save_name", default="processed")

    args = parse.parse_args()

    ## begin count img

    img_list = []
    
    data_dirs = glob.glob("../../data/Tanimoto_eLife_Fig3B/*")
    
    for dir_i in data_dirs:
        img_list.extend(glob.glob(dir_i + "/main/*"))
    
    START_ID, END_ID = len(img_list) // 4 * args.process_id, len(img_list) // 4* (args.process_id + 1)
    
    print("load data from ", START_ID, "to", END_ID, "all:", len(img_list))
    del img_list

    ## end count img

    loader = load_datasets(START_ID, END_ID)

    init_t = time.time()

    count_delete_img = 0

    for data_i, (data, date) in enumerate(loader):
        data = data[0]
        date = date[0]
        dir_i = "../../data/" + args.save_name + "/" + date
        #print("tensor:", data.shape)

        if len(data.shape) != 4:
            print(data_i + START_ID , "/", END_ID, " Not save because of fail to load")
            count_delete_img += 1

        elif torch.sum(data) == 0:
            print(data_i + START_ID , "/", END_ID, " Not save because of celegans on edge")
            count_delete_img += 1

        else:
            os.makedirs(dir_i, exist_ok=True)
            torch.save(data, dir_i + "/tensor_{:0=6}.pt".format(data_i + START_ID))

        if (data_i + START_ID) % 1000 == 0:
            print(data_i + START_ID , "/", END_ID, " Load&Save Processd : ", time.time() - init_t)

    print("TOTAL DELETE:", count_delete_img)
    print(data_i + START_ID , "/", END_ID, " Finish Processd : ", time.time() - init_t)
