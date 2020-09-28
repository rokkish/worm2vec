"""run run_worm2vec.py with multi-dataset"""
import subprocess
import glob
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
    data = glob.glob("/root/worm2vec/data/variety_data_r36_n50_np/*.npz")
    data = sorted(data)
    if len(data) == 0:
        raise ValueError("data not found")

    epoch = 10

    for i, data_i in enumerate(data):

        logger.info("train: {}/{}".format(i+1, len(data)))
        post("train: {}/{}".format(i+1, len(data)))

        # skip
        if i in []:
            prev_date = glob_prev_datetime()
            continue

        # run
        if i == 0:
            subprocess.call(["python", "run_worm2vec.py",
                "path.worm_data={}".format(data_i),
                "nn.n_epochs={}".format(epoch),
                "nn.batch_size=1",
                "train.restart_train=False",
                "nn.n_negative=50",
            ])
            prev_date = glob_prev_datetime()
        # reload params
        else:
            subprocess.call(["python", "run_worm2vec.py",
                "path.worm_data={}".format(data_i),
                "nn.n_epochs={}".format(epoch),
                "nn.batch_size=1",
                "train.restart_train=True",
                "path.checkpoint_fullpath={}/checkpoints/model.ckpt".format(prev_date),
                "nn.n_negative=50",
            ])
            prev_date = glob_prev_datetime()
