"""
copy date/tensor_000000.pt date_000000.pt
"""
import torch
import glob
import time


def copy_rename(path_list, root_dir):
    init_time = time.time()
    for idx, path in enumerate(path_list):
        tmp = path.split(root_dir)[1].split("/")
        date, tensor_id = tmp[0], tmp[1]
        dataid = tensor_id.split("_")[1]
        tensor = torch.load(path)
        torch.save(tensor, "{}all_data/{}_{}".format(root_dir, date, dataid))
        if idx % 1000 == 0:
            print("end: ", idx, time.time() - init_time, "[sec]")


def get_path_list(root_dir):
    path_list = []
    data_dirs = sorted(glob.glob(root_dir + "*"))

    for dir_i in data_dirs:
        if "all_data" in dir_i:
            continue
        path_list.extend(sorted(glob.glob(dir_i + "/*")))

    print(len(path_list))

    return path_list


def main():
    root_dir = "../../../data/processed/"
    path_list = get_path_list(root_dir)
    copy_rename(path_list, root_dir)


if __name__ == "__main__":
    main()
