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
        output_data = meta_datas[meta_datas['symbol'] == key].drop('symbol', axis=1, inplace=False)
        date = output_data['date'].apply(lambda x: x.replace("-", ""))
        output_data = output_data.drop('date', axis=1, inplace=False)
        output_data = output_data.rename(index=date)
        output_data.to_csv(save_path + key + ".csv")


if __name__ == "__main__":
    split_datas("prices-split-adjusted.csv")
