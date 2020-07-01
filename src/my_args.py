
import argparse
import config

parse = argparse.ArgumentParser(description="Worm2Vec")

parse.add_argument("-e", "--epoch", type=int, default=config.epoch)

parse.add_argument("--logdir", type=str, default=config.logdir, help="set path of logfile ../log/tensorboard/[logdir]")

parse.add_argument("--gpu_id", type=str, default=config.gpu_id,
            help="When you want to use 1 GPU, input 0. Using Multiple GPU, input [0, 1]")

parse.add_argument("--traindir", type=str, default=config.train_dir, help="set path of train data dir ../../data/[traindir]")

parse.add_argument("--use_rotate", action="store_true", \
    help="if True, train with rotate data(theta=0, 10, ..350), else train with only original data(theta=0)")

parse.add_argument("--model_name", type=str, default=config.model_name, help="Set Name of model ../models/[model_name]")

parse.add_argument("--model", type=str, default=config.model, help="Set Type of model. (cbow, vae)")

parse.add_argument("--loss_function_name", type=str, default=config.loss_function_name, help="Set Type of loss.")

parse.add_argument("--zsize", type=int, default=config.zsize, help="Set shape of encoded.")

parse.add_argument("--sequential", action="store_true", \
    help="if True use sequential data, else non sequential data.")

parse.add_argument("--reverse", action="store_true", \
    help="if True model maximize loss")


# predict.py
parse.add_argument("--test_shuffle", action="store_true")

parse.add_argument("--max_predict", type=int, default=config.max_predict)

parse.add_argument("--num_of_tensor_to_embed", type=int, default=config.num_of_tensor_to_embed)

# analyze.py
parse.add_argument("--max_analyze", type=int, default=config.max_analyze)

args = parse.parse_args()
