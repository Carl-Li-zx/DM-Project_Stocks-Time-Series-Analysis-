import pandas as pd
from tqdm import tqdm
import os


def split_datas(file_path):
    save_path = "../dataset/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    meta_datas = pd.read_csv(file_path)
    meta_datas["volume"] = meta_datas["volume"] * 1e-7
    keys = meta_datas['symbol'].unique().tolist()
    for key in tqdm(keys, total=len(keys)):
        meta_datas[meta_datas['symbol'] == key].to_csv(save_path + key + ".csv", index=False)


split_datas("../meta_data/prices-split-adjusted.csv")
