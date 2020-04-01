import logging

def get_logger(name, level=logging.DEBUG, save_name="../log/logger/test.log"):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # create console handler and set level to debug
    sh = logging.StreamHandler()
    fh = logging.FileHandler(save_name)
    sh.setLevel(logging.INFO)
    fh.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s')

    # add formatter to handler
    sh.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add handler to logger
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger
