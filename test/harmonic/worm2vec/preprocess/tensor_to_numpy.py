"""
how to run
    test/harmonic/worm2vec# python preprocess/tensor_to_numpy.py --load_path ../../../../data/processed/varietydata_r36_n5_1_3 --save_path ../../../../data/variety_r36_n5/np_1_3 --datasize 1000

"""
import os
import torch
import numpy as np
import glob
import argparse


def main(args):
    os.makedirs(args.save_path, exist_ok=True)
    paths = sorted(glob.glob(args.load_path + "/*"))
    max_processes = len(paths)//args.datasize + 1
    history_np = sorted(glob.glob(args.save_path + "/*"))
    print(len(paths), len(history_np))

    for process in range(max_processes):
        # 処理済みであればskip
        if "{}/{:0=2}.npz".format(args.save_path, process) in history_np:
            print("skip {}".format(process))
            continue

        # pathsは125710なので，以下で場合分け
        if process == max_processes - 1:
            now_paths = paths[args.datasize * process:]
        else:
            now_paths = paths[args.datasize * process: args.datasize * (process + 1)]

        sample_tensor = torch.load(now_paths[0])
        shapes = [len(now_paths)] + list(sample_tensor.shape)
        arr = np.zeros(shapes)

        for i, path in enumerate(now_paths):
            t = torch.load(path)
            t_np = t.numpy()
            arr[i] = t_np
            print("\r [{}] {}/{}".format(process, i+1, shapes[0]), end="")

        np.savez_compressed("{}/{:0=3}".format(args.save_path, process), arr)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--load_path", default="")
    parse.add_argument("--save_path", default="")
    parse.add_argument("--datasize", type=int, default=10000)
    args = parse.parse_args()
    main(args)
