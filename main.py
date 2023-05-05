from src.config import Config
from src.dataset import *
from src.model import *
from src.trainer import *
from src.utils import *


def main(config: Config):
    logger = load_logger(config)
    try:
        np.random.seed(config.random_seed)  # 设置随机种子，保证可复现

        if config.do_train:
            train(config, logger)
    except Exception:
        logger.error("Run Error", exc_info=True)


if __name__ == "__main__":
    import argparse

    # argparse方便于命令行下输入参数，可以根据需要增加更多
    parser = argparse.ArgumentParser()
    # parser.add_argument("-t", "--do_train", default=False, type=bool, help="whether to train")
    # parser.add_argument("-p", "--do_predict", default=True, type=bool, help="whether to train")
    # parser.add_argument("-b", "--batch_size", default=64, type=int, help="batch size")
    # parser.add_argument("-e", "--epoch", default=20, type=int, help="epochs num")
    args = parser.parse_args()

    con = Config()
    for key in dir(args):  # dir(args) 函数获得args所有的属性
        if not key.startswith("_"):  # 去掉 args 自带属性，比如__name__等
            setattr(con, key, getattr(args, key))  # 将属性值赋给Config

    main(con)
