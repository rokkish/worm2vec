"""run run_worm2vec.py with multi-dataset"""
import os
import datetime
import subprocess
import glob
from distutils.dir_util import copy_tree
from post_slack import post
import get_logger
logger = get_logger.get_logger(name="multi_run")


def glob_prev_datetime():
    """
    Returns:
        [str]: "2020-09-01/00-00-00"
    """
    root = "/root/worm2vec/worm2vec/test/harmonic/worm2vec/outputs"
    date_dir = sorted(glob.glob(root + "/*"))
    prev_date_dir = date_dir[-1]
    time_dir = sorted(glob.glob(prev_date_dir + "/*"))
    prev_datetime_dir = time_dir[-1]

    logger.debug(prev_datetime_dir)

    return prev_datetime_dir


if __name__ == "__main__":

    logger.info("prev dir: {}".format(glob_prev_datetime()))

    # glob data
    data = glob.glob("/root/worm2vec/data/variety_data_strict_r36_n100_np/*.npz")
    data = sorted(data)
    if len(data) == 0:
        raise ValueError("data not found")

    epoch = 12
    n_classes = 10
    n_negative = 70
    fixedtestdata = "/root/worm2vec/data/variety_data_strict_r36_n100_np/minitest/000_2000.npz"
    exp_name = "fulltrain neegative_top70"
    exp_name_ = "plt " + exp_name

    runned_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    os.makedirs("/root/worm2vec/worm2vec/test/harmonic/worm2vec/logs/test_score/{}".format(runned_datetime))

    for i, data_i in enumerate(data):

        logger.info("train: {}/{}".format(i+1, len(data)))
        post("train: {}/{}".format(i+1, len(data)))

        # skip
        if i in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
            prev_date = glob_prev_datetime()
            continue

        # run
        if i == 0:
            subprocess.call(["python", "run_worm2vec.py",
                "path.worm_data={}".format(data_i),
                "path.fixedtestdata={}".format(fixedtestdata),
                "nn.n_epochs={}".format(epoch),
                "nn.batch_size=1",
                "train.restart_train=False",
                "nn.n_negative={}".format(n_negative),
                "path.test_score=/root/worm2vec/worm2vec/test/harmonic/worm2vec/logs/test_score/{}/cossim_{:0=2}.csv".format(runned_datetime, i),
                "nn.n_classes={}".format(n_classes),
                "exp_name={}".format(exp_name),
            ])
            prev_date = glob_prev_datetime()
        # reload params
        else:
            subprocess.call(["python", "run_worm2vec.py",
                "path.worm_data={}".format(data_i),
                "path.fixedtestdata={}".format(fixedtestdata),
                "nn.n_epochs={}".format(epoch),
                "nn.batch_size=1",
                "train.restart_train=True",
                "path.checkpoint_fullpath={}/checkpoints/model.ckpt".format(prev_date),
                "nn.n_negative={}".format(n_negative),
                "path.test_score=/root/worm2vec/worm2vec/test/harmonic/worm2vec/logs/test_score/{}/cossim_{:0=2}.csv".format(runned_datetime, i),
                "nn.n_classes={}".format(n_classes),
                "exp_name={}".format(exp_name),
            ])
            prev_date = glob_prev_datetime()

        # predict
        """
        subprocess.call(["python", "run_worm2vec.py",
            "path.worm_data={}".format(data_i),
            "path.fixedtestdata={}".format(fixedtestdata),
            "path.checkpoint_fullpath={}/checkpoints/model.ckpt".format(prev_date),
            "nn.n_negative={}".format(n_negative),
            "nn.batch_size=1",
            "path.tensorboard=./",
            "train_mode=False",
            "nn.n_classes={}".format(n_classes),
            "exp_name={}".format(exp_name_),
        ])
        predict_date = glob_prev_datetime()

        # to load model.ckpt next iter
        load_checkpoints_dir = "{}/checkpoints/".format(prev_date)
        cp_checkpoints_dir = "{}/checkpoints/".format(predict_date)
        copy_tree(load_checkpoints_dir, cp_checkpoints_dir)
        """
