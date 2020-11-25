import subprocess
import get_logger
logger = get_logger.get_logger(name='make_prepro')
from preprocess import post

if __name__ == '__main__':

    root_dir = "processed4"

    post("start make_prepro.py")

    """
    try:
        ret = subprocess.call(["python", "preprocess.py",
            "--process_id", "0",
            "--save_name", root_dir,
            "--root_dir", "../../data/raw/unpublished_control"
        ])
    except:
        post("error happen@preprocess.py")
        raise ValueError("Break")
    else:
        post("error not happen@preprocess.py")
    finally:
        post(ret)
    try:
        ret = subprocess.call(["python", "features/rename.py",
            "--root_dir", "../../data/{}/".format(root_dir)
        ])
    except:
        post("error happen@rename.py")
        raise ValueError("Break")
    else:
        post("error not happen@rename.py")
    finally:
        post(ret)

    #"""
    try:
        ret = subprocess.call(["python", "get_distance_table.py",
            "--process_id", "0",
            "--max_pair", "200",
            "--max_original", "100000",
            "--gpu_id", "0",
            "--root_dir", "../../data/{}".format(root_dir),
        ])
    except:
        post("error happen@get_distance_table.py")
        raise ValueError("Break")
    else:
        post("error not happen@get_distance_table.py")
    finally:
        post(ret)

    try:
        ret = subprocess.call(["python", "compress_distance_table.py",
            "-K", "100",
            "--root_dir", "../../data/{}".format(root_dir)
        ])
    except:
        post("error happen@compress_distance_table.py")
        raise ValueError("Break")
    else:
        post("error not happen@compress_distance_table.py")
    finally:
        post(ret)

    try:
        ret = subprocess.call(["python", "make_variety_dataset.py",
            "--load_K", "100",
            "--num_rotate", "36",
            "--num_negative", "100",
            "--save_path", "varietydata_r36_n100",
            "--root_dir", "../../data/{}/".format(root_dir),
        ])
    except:
        post("error happen@make_variety_dataset.py")
        raise ValueError("Break")
    else:
        post("error not happen@make_variety_dataset.py")
    finally:
        post(ret)


    try:
        ret = subprocess.call(["python", "../test/harmonic/worm2vec/preprocess/tensor_to_numpy.py",
            "--load_path", "../../data/{}/varietydata_r36_n100".format(root_dir),
            "--save_path", "../../data/{}/np_n100".format(root_dir),
            "--datasize", "10000",
        ])
    except:
        post("error happen@tensor_to_numpy.py")
        raise ValueError("Break")
    else:
        post("error not happen@tensor_to_numpy.py")
    finally:
        post(ret)
    post("finish make_prepro.py")

