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

import config
from features.worm_transform import ToBinary, FillHole, Labelling, Padding, ToNDarray

STAR_ID = 14580

class WormDataset_prepro(torch.utils.data.Dataset):
    """
        clone from "class WormDataset"
        diff:
            Load all of raw data to preprocess.
    """

    def __init__(self, root, transform=None):

        self.root = root    # root_dir \Tanimoto_eLife_Fig3B or \unpublished control
        self.transform = transform
        self.data = []

        data_dirs_all = glob.glob(self.root + "/*")
        data_dirs = data_dirs_all

        for dir_i in data_dirs:
            self.data.extend(glob.glob(dir_i + "/main/*"))
        self.data = self.data[STAR_ID:]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image)
        """
        img = self.data[index]

        try:
            img = Image.open(img)
        except IOError:
            print("IOError")
            return 0

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.data)

def load_datasets():
    """ Set transform """
    worm_transforms = transforms.Compose([
        ToBinary(),
        FillHole(),
        Labelling(),
        Padding(),
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor()])

    """ Set dataset """
    dataset = WormDataset_prepro(root="../../data/Tanimoto_eLife_Fig3B",
        transform=worm_transforms)

    """ Dataloader """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False)

    return loader

if __name__ == "__main__":

    loader = load_datasets()
    init_t = time.time()

    for data_i, data in enumerate(loader):
        if len(data.shape) == 4:
            torch.save(data, "../../data/processed/tensor_{:0=10}.pt".format(data_i + STAR_ID))
        else:
            print(data_i + STAR_ID , "/", 259913, " Not save")

        if (data_i + STAR_ID) %1000==0:
            print(data_i + STAR_ID , "/", 259913, " Load&Save Processd : ", time.time() - init_t)

    print(data_i + STAR_ID , "/", 259913, " Finish Processd : ", time.time() - init_t)
