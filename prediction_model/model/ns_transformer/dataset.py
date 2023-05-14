import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from prediction_model.model.ns_transformer.config import Config


class Data:
    def __init__(self, config: Config, file_name: str):
        self.config = config
        path = self.config.train_data_path + file_name
        self.data, self.data_column_name = self.read_data(path)

        self.data_num = self.data.shape[0]
        self.train_num = int(self.data_num * self.config.train_data_rate)

    def read_data(self, path):
        init_data = pd.read_csv(path, usecols=self.config.feature_columns, index_col=0)
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
    def __init__(self, datas: pd.DataFrame, test_start_date, field):
        super(TestDataset, self).__init__()
        self.config = Config()

        columns_idx = datas.columns.to_list()
        self.pred_idx = columns_idx.index(field)

        self.train_data, self.test_data = datas[pd.to_datetime(datas.index.astype(str)) < pd.to_datetime(
            test_start_date)], datas[pd.to_datetime(datas.index.astype(str)) >= pd.to_datetime(test_start_date)]
        self.labels = self.test_data[field].values.tolist()
        self.dates = pd.to_datetime(self.test_data.index.astype(str)).strftime('%Y-%m-%d').tolist()
        seq = self.train_data[-self.config.seq_len:]
        self.test_data = pd.concat([seq, self.test_data], axis=0)
        self.test_x = []
        self.test_y = []
        self.test_data_value = self.test_data.values.tolist()
        for i in range(len(self.test_data) - self.config.seq_len - self.config.pred_len + 1):
            self.test_x.append(self.test_data_value[i:i+self.config.seq_len])
            self.test_y.append(self.test_data_value[i+self.config.seq_len-self.config.label_len:
                                                    i+self.config.seq_len+self.config.pred_len])

    def __getitem__(self, index) -> T_co:
        return torch.tensor(self.test_x[index]), torch.tensor(self.test_y[index])

    def __len__(self):
        return len(self.test_x)
