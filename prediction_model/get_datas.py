from data_api import api, constant
import pandas as pd
import numpy as np


def get_data(code, field, start_time, end_time):
    field = field.lower()
    assert field in constant.SUPPORT_FIELDS, f"field {field} should in {constant.SUPPORT_FIELDS}"
    pd_data = get_candlestick_chart_data(code, start_time, end_time)
    if pd_data is not None:
        dates = pd.to_datetime(pd_data.index.astype(str)).strftime('%Y-%m-%d')
        dates = np.array([dates]).transpose((1, 0))
        values = pd_data.values
        o = np.concatenate((dates, values), axis=1)
        pd_data["volume"] = pd_data["volume"] * 1e-7

        return o.tolist(), pd_data
    else:
        return None, None


def get_candlestick_chart_data(code, start_time, end_time):
    pd_data = None
    if code in constant.SUPPORT_FIELDS:
        pd_data = api.get_stock_data(code, start_time, end_time)
    elif code in constant.NEW_YORK_STOCK_CODE:
        import pandas as pd
        pd_data = pd.read_csv(f"dataset/{code}.csv", index_col=0)
        pd_data["volume"] = pd_data["volume"] * 1e7
    if pd_data is not None:
        pd_data = pd_data[
            (pd.to_datetime(end_time) > pd.to_datetime(pd_data.index.astype(str))) &
            (pd.to_datetime(pd_data.index.astype(str)) >= pd.to_datetime(start_time))
            ]
    return pd_data
