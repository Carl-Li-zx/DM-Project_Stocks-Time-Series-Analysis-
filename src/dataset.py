import pandas as pd
from tqdm import tqdm

FILE_NAME = "../dataset/prices-split-adjusted.csv"


class PriceDataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.meta_datas = None
        self.datas = None
        self._read()
        self._split_datas()

    def _read(self):
        self.meta_datas = pd.read_csv(self.file_path)

    def _split_datas(self):
        keys = self.meta_datas['symbol'].unique().tolist()
        self.datas = {}
        for key in tqdm(keys, total=len(keys)):
            self.datas[key] = self.meta_datas[self.meta_datas['symbol'] == key]

    def get_keys(self):
        return self.datas.keys()

    def get_item(self, key):
        return self.datas[key]
