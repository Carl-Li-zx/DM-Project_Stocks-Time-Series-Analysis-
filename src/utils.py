import logging
import matplotlib
import numpy as np
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
import torch

matplotlib.use('Agg')

from src.config import Config
from src.dataset import Data


def load_logger(config):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    file_handler = RotatingFileHandler(config.log_save_path + "log.txt", maxBytes=1024000, backupCount=5)
    file_handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    config_dict = {}
    for key in dir(config):
        if not key.startswith("_"):
            config_dict[key] = getattr(config, key)
    config_str = str(config_dict)
    config_list = config_str[1:-1].split(", '")
    config_save_str = "\nConfig:\n" + "\n'".join(config_list)
    logger.info(config_save_str)

    return logger


def draw(config: Config, origin_data: Data, logger, predict_norm_data: np.ndarray):
    label_data = origin_data.data[origin_data.train_num + origin_data.start_num_in_test:,
                 config.label_in_feature_index]
    predict_data = predict_norm_data * origin_data.std[config.label_in_feature_index] + \
                   origin_data.mean[config.label_in_feature_index]  # 通过保存的均值和方差还原数据
    assert label_data.shape[0] == predict_data.shape[0], "The element number in origin and predicted data is different"

    label_name = [origin_data.data_column_name[i] for i in config.label_in_feature_index]
    label_column_num = len(config.label_columns)

    # label 和 predict 是错开config.predict_day天的数据的
    # 下面是两种norm后的loss的计算方式，结果是一样的，可以简单手推一下
    # label_norm_data = origin_data.norm_data[origin_data.train_num + origin_data.start_num_in_test:,
    #              config.label_in_feature_index]
    # loss_norm = np.mean((label_norm_data[config.predict_day:] - predict_norm_data[:-config.predict_day]) ** 2, axis=0)
    # logger.info("The mean squared error of stock {} is ".format(label_name) + str(loss_norm))

    loss = np.mean((label_data[config.predict_day:] - predict_data[:-config.predict_day]) ** 2, axis=0)
    loss_norm = loss / (origin_data.std[config.label_in_feature_index] ** 2)
    logger.info("The mean squared error of stock {} is ".format(label_name) + str(loss_norm))

    label_X = range(origin_data.data_num - origin_data.train_num - origin_data.start_num_in_test)
    predict_X = [x + config.predict_day for x in label_X]

    for i in range(label_column_num):
        plt.figure(i + 1)  # 预测数据绘制
        plt.plot(label_X, label_data[:, i], label='label')
        plt.plot(predict_X, predict_data[:, i], label='predict')
        plt.title("Predict stock {} price".format(label_name[i], ))
        logger.info("The predicted stock {} for the next {} day(s) is: ".format(label_name[i], config.predict_day) +
                    str(np.squeeze(predict_data[-config.predict_day:, i])))
        if config.do_figure_save:
            plt.savefig(
                config.figure_save_path + "{}predict_{}.png".format(config.continue_flag, label_name[i]))

        # plt.show()


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, logger):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, logger)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, logger)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, logger):
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + 'checkpoint.pth')
        self.val_loss_min = val_loss
