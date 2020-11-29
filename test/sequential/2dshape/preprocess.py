"""2dshape画像から画素平均を作成，npファイルで保存
"""
import os
from PIL import Image
import numpy as np


class Preprocess(object):
    def __init__(self):
        self.data_path = "./data/raw/"
        self.np_path = "./data/raw_np/"
        self.save_path = "./data/processed/"
        self.data = {}
        self.fig = ["circle", "square", "triangle"]
        self.min_data_length = 10**4
        self.resize_width = 32
        self.resize_height = 32
        # flag
        self.translate_into_np_is_needed = False

    def run(self, parameter_list=None):
        """
        Steps.
        * Load data
        * Translate into np file
        * Create pair of different images
        * Compute mean pair
        * Save result
        """
        self.load_data()
        self.translate_into_np()
        if not self.translate_into_np_is_needed:
            self.min_data_length = 1000
            self.match_data_length()
            self.create_pair_list()
            self.compute_mean_pair()

    def get_file_number(self, val):
        """
        Args:
            val (path fo bmp file)  | ~/data/raw/fig/0.(png|npy)
        Return:
            file_number (int)       | 0
        """
        file_number = val.split("/")[-1].split(".")[0]

        return int(file_number)

    def load_data(self, parameter_list=None):
        import glob
        for fig_i in self.fig:
            if self.translate_into_np_is_needed:
                ls = glob.glob(self.data_path + fig_i + "/*.png")
            else:
                ls = glob.glob(self.np_path + fig_i + "/*.npy")
            self.data[fig_i] = sorted(ls, key=self.get_file_number)
            self.min_data_length = min(len(ls), self.min_data_length)

    def resize(self, img):
        return img.resize((self.resize_width, self.resize_height))

    def translate_into_np(self):
        if self.translate_into_np_is_needed:

            for fig_i in self.fig:

                os.makedirs(os.path.join(self.np_path, fig_i), exist_ok=True)

                for j, path in enumerate(self.data[fig_i]):
                    img = Image.open(path)
                    img = self.resize(img)
                    img = np.array(img)
                    np.save("{}{}/{}".format(self.np_path, fig_i, str(j)), img)
                    print("\r {} {}/{}".format(fig_i, j, len(self.data[fig_i])), end="")

    def match_data_length(self, parameter_list=None):
        for fig_i in self.fig:
            self.data[fig_i] = self.data[fig_i][:self.min_data_length]

    def create_pair_list(self, parameter_list=None):
        import itertools
        self.pair_list_figures = itertools.combinations(self.fig, 2)

    def compute_mean_pair(self, parameter_list=None):
        for (fig_i, fig_j) in self.pair_list_figures:
            arr_j = np.zeros((self.min_data_length, 11, self.resize_width, self.resize_height))
            os.makedirs("{}{}_{}".format(self.save_path, fig_i, fig_j), exist_ok=True)

            for j, path_j in enumerate(self.data[fig_j]):
                arr_j[j] += np.load(path_j)

            for i, path_i in enumerate(self.data[fig_i]):
                arr = np.zeros((self.min_data_length, 11, self.resize_width, self.resize_height))
                np_i = np.load(path_i)
                for beta in range(0, 11):
                    arr[:, beta] += (1 - beta * 0.1) * np_i
                    arr[:, beta] += beta * 0.1 * arr_j[:, beta]
                arr /= 2
                np.save("{}{}_{}/{:0=4}".format(self.save_path, fig_i, fig_j, i), arr)
                print("\r {}_{} {}/{}".format(fig_i, fig_j, i, len(self.data[fig_i])), end="")


def main():
    preprocessor = Preprocess()
    preprocessor.run()


if __name__ == "__main__":
    main()
