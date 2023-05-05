import os
import random

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from .config import Config


class Data:
    def __init__(self, config: Config, file_name: str):
        self.config = config
        path = self.config.train_data_path + file_name
        self.data, self.data_column_name = self.read_data(path)

        self.data_num = self.data.shape[0]
        self.train_num = int(self.data_num * self.config.train_data_rate)

    def read_data(self, path):
        init_data = pd.read_csv(path, usecols=self.config.feature_columns)
        return init_data.values, init_data.columns.tolist()

    def get_train_data(self):
        assert self.train_num + self.config.pred_len < self.data_num
        train_x = [self.data[i:i + self.config.seq_len] for i in range(self.train_num - self.config.seq_len + 1)]
        train_y = [self.data[i + self.config.seq_len - self.config.label_len:
                             i + self.config.seq_len + self.config.pred_len]
                   for i in range(self.train_num - self.config.seq_len + 1)]
        return np.array(train_x), np.array(train_y)

    def get_valid_data(self):
        valid_x = [self.data[i:i + self.config.seq_len] for i in
                   range(self.train_num, self.data_num - self.config.seq_len - self.config.pred_len + 1)]
        valid_y = [self.data[i + self.config.seq_len - self.config.label_len:
                             i + self.config.seq_len + self.config.pred_len]
                   for i in range(self.train_num, self.data_num - self.config.seq_len - self.config.pred_len + 1)]
        return np.array(valid_x), np.array(valid_y)

    def get_test_data(self):
        test_x = [self.data[i:i + self.config.seq_len] for i in
                  range(0, self.data_num - self.config.seq_len - self.config.pred_len + 1)]
        test_y = [self.data[i + self.config.seq_len - self.config.label_len:
                            i + self.config.seq_len + self.config.pred_len]
                  for i in range(0, self.data_num - self.config.seq_len - self.config.pred_len + 1)]
        return np.array(test_x), np.array(test_y)


class StockDataset(Dataset):
    def __init__(self, config, state="train"):
        super(StockDataset, self).__init__()
        self.config = config
        self.input_datas, self.labels = None, None
        for file in os.listdir(config.train_data_path):
            data_provider = Data(config, file)
            i, l = data_provider.get_train_data() if state == "train" else data_provider.get_valid_data()
            try:
                if self.labels is None:
                    self.input_datas, self.labels = i, l
                else:
                    self.input_datas = np.concatenate((self.input_datas, i), axis=0)
                    self.labels = np.concatenate((self.labels, l), axis=0)
                    # print(f"file: {file}, i:{i.shape}, input_datas: {self.input_datas.shape}")
            except Exception:
                print(f"Exception\tfile: {file}, i:{i.shape}, input_datas: {self.input_datas.shape}")
                continue

    def __getitem__(self, index) -> T_co:
        return self.input_datas[index], self.labels[index]

    def __len__(self):
        return len(self.input_datas)


class TestDataset(Dataset):
    def __init__(self, config, test_file=None):
        super(TestDataset, self).__init__()
        self.config = config
        if test_file is None:
            files = list(os.listdir(self.config.train_data_path))
            test_file = random.sample(files, 1)[0]
        self.data_provider = Data(self.config, test_file)
        self.input_datas, self.labels = self.data_provider.get_test_data()

    def __getitem__(self, index) -> T_co:
        return self.input_datas[index], self.labels[index]

    def __len__(self):
        return len(self.input_datas)
