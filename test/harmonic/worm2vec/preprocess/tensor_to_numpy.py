"""
how to run
    test/harmonic/worm2vec# python tensor_to_numpy.py --process 0~6

"""
import torch
import numpy as np
import glob
import argparse


def main(args):
    paths = sorted(glob.glob("../../../../data/varietydata_r36_n5/*"))

    # pathsの最大は約63000なので，以下で場合分け
    if args.process == 6:
        paths = paths[10000 * args.process:]
    elif args.process > 6:
        return
    else:
        paths = paths[10000 * args.process: 10000 * (args.process + 1)]
    sample_tensor = torch.load(paths[0])
    shapes = [len(paths)] + list(sample_tensor.shape)
    arr = np.zeros(shapes)

    for i, path in enumerate(paths):
        t = torch.load(path)
        t_np = t.numpy()
        arr[i] = t_np
        print("\r {}/{}".format(i+1, shapes[0]), end="")

    np.savez_compressed("../../../../data/varietydata_r36_n5_{}".format(args.process), arr)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--process", type=int, default=0)
    args = parse.parse_args()
    main(args)
