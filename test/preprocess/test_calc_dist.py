"""
    To calculate distance between ImageA and ImageB
"""
import argparse
import pandas as pd
import numpy as np
import glob


def count_img(process_id):
    """Count num dataset, and return (start, end) id to divide data. 
        Arg:
            process_id (int): No.[0, 1, 2, 3] of docker container.
    """

    img_list = glob.glob("../../../data/processed/alldata/*")

    START_ID, END_ID = len(img_list) // 4 * process_id, \
        len(img_list) // 4 * (process_id + 1)

    return START_ID, END_ID


def main(args):
    """Load datasets, Do preprocess()
    """
    START_ID, END_ID = count_img(args.process_id)
    print(START_ID, END_ID)

#    loader = load_datasets(START_ID, END_ID)

#    preprocess(START_ID, END_ID, loader, args.process_id, args.test)

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--process_id", type=int)
    parse.add_argument("--test", action="store_true")
    args = parse.parse_args()
    main(args)

