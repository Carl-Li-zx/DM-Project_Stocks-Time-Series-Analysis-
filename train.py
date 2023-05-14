import argparse
import numpy as np

from prediction_model.model.ns_transformer.config import Config
from prediction_model.model.ns_transformer.trainer import train
from prediction_model.model.utils import load_logger


def train_ns_transformer():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", default=1024, type=int, help="batch size")
    parser.add_argument("-e", "--epoch", default=50, type=int, help="epochs num")
    args = parser.parse_args()
    config = Config()
    for key in dir(args):
        if not key.startswith("_"):
            setattr(config, key, getattr(args, key))

    logger = load_logger(config)
    try:
        np.random.seed(config.random_seed)
        if config.do_train:
            train(config, logger)
    except Exception:
        logger.error("Run Error", exc_info=True)


if __name__ == "__main__":
    train_ns_transformer()
