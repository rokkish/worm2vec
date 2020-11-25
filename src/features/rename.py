"""
copy date/tensor_000000.pt date_000000.pt
"""
import os
import torch
import glob
import time
import argparse

def copy_rename(path_list, root_dir):
    init_time = time.time()
    for idx, path in enumerate(path_list):
        tmp = path.split(root_dir)[1].split("/")
        date, tensor_id = tmp[0], tmp[1]
        dataid = tensor_id.split("_")[1]
        tensor = torch.load(path)
        torch.save(tensor, "{}alldata/{}_{}".format(root_dir, date, dataid))
        if idx % 1000 == 0:
            print("\rend: {} {}[sec]".format(idx, time.time() - init_time), end="")


def get_path_list(root_dir):
    path_list = []
    data_dirs = sorted(glob.glob(root_dir + "*"))
    print(data_dirs[0])

    for dir_i in data_dirs:
        if "alldata" in dir_i:
            continue
        path_list.extend(sorted(glob.glob(dir_i + "/*.pt")))

    print(len(path_list), path_list[0])

    return path_list


def main(args):
    os.makedirs("{}alldata".format(args.root_dir), exist_ok=True)
    path_list = get_path_list(args.root_dir)
    copy_rename(path_list, args.root_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="../../data/processed/")
    args = parser.parse_args()
    main(args)
