import logging
import matplotlib
import numpy as np
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

matplotlib.use('Agg')

from prediction_model.model.ns_transformer.config import Config


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


def draw(config: Config, result: np.ndarray, test_dataset: np.ndarray, feature_index, figure_save_name):
    ground_truth = test_dataset[:, feature_index]
    predict = predict_upper = predict_lower = ground_truth[0:config.seq_len]
    result = result[:, :, feature_index]
    result1 = np.append(result[:, 0], result[-1, 1])
    result2 = np.append(result[0, 0], result[:, 1])
    max_r = np.max([result1, result2], axis=0)
    min_r = np.min([result1, result2], axis=0)
    avg = (max_r + min_r) / 2
    predict = np.concatenate((predict, avg), axis=0)
    predict_lower = np.concatenate((predict_lower, min_r), axis=0)
    predict_upper = np.concatenate((predict_upper, max_r), axis=0)

    x = list(range(len(ground_truth)))
    plt.plot(x, ground_truth, label='Ground Truth')
    plt.plot(x, predict, label='Predict Avg.')
    plt.fill_between(x, predict_upper, predict_lower, alpha=0.5, label='Error')

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(config.figure_save_path + figure_save_name)

    mae = mean_absolute_error(ground_truth, predict)
    mse = mean_squared_error(ground_truth, predict)
    return mae, mse


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, logger, model_name):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, logger, model_name)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, logger, model_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, logger, model_name):
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + model_name)
        self.val_loss_min = val_loss
