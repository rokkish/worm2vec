import os
import glob
import numpy as np


def load_morph_data():
    dic = {}
    dirs = glob.glob("./processed/morph/*")

    for d in dirs:
        files = glob.glob(d + "/*.npy")
        dir_name = d.split("/")[-1]
        dic[dir_name] = files

    print(dic.keys())

    return dic


def slice_save_data(data):
    for key in data.keys():

        dir_path = "./processed/minimorph/{}".format(key)
        os.makedirs(dir_path, exist_ok=True)

        for i, path in enumerate(data[key]):

            save_path = "./processed/minimorph/{}/{:0=5}".format(key, i)

            arr = np.load(path)

            if i == 0:
                print(arr.shape)
                if arr.shape != (11, 64, 64):
                    assert("dont load morph data")

            sliced_arr = np.zeros((3, 64, 64))
            sliced_arr[0] = arr[0]
            sliced_arr[1] = arr[5]
            sliced_arr[2] = arr[-1]

            np.save(save_path, sliced_arr)

            print("\r [{:0=5}/{:0=5}]".format(i, len(data[key]),), end="")


def preprocess():
    data = load_morph_data()
    slice_save_data(data)


def main():
    """slice morph data
        N * (11, w, h) -> N * (3, w, h)
    """
    preprocess()


if __name__ == "__main__":
    main()
