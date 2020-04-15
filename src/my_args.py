
import argparse
parse = argparse.ArgumentParser(description="Worm2Vec")

parse.add_argument("-e", "--epoch", type=int, default=15)

parse.add_argument("--logdir", type=str, default="default", help="set path of logfile ../log/tensorboard/[logdir]")

parse.add_argument("--gpu_id", type=str, default="0",
            help="When you want to use 1 GPU, input 0. Using Multiple GPU, input [0, 1]")

parse.add_argument("--traindir", type=str, default="processed/201302081353", help="set path of train data dir ../../data/[traindir]")

parse.add_argument("-w", "--window", type=int, default=3)

parse.add_argument("--use_rotate", action="store_true", \
    help="if True, train with rotate data(theta=0, 10, ..350), else train with only original data(theta=0)")

parse.add_argument("--model_name", type=str, default="VAEmodel", help="Set Name of model ../models/[model_name]")

# predict.py
parse.add_argument("--max_predict", type=int, default=3)

args = parse.parse_args()
