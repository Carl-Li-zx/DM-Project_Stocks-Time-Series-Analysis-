from prediction_model.model.ns_transformer.trainer import *
from prediction_model.model.utils import *
import argparse
import os
from data_api.constant import CODE_LIST


class Share:
    stock_code = None
    field = None
    shared_data = None
    training_start_date = None
    test_start_date = None
    test_end_date = None
    all_data = None
    codes = CODE_LIST


def prediction(data, startdate, model="arima", field='close'):
    if model.lower() == "arima":
        from prediction_model.model.arima.arima_model import prediction as pred_func
    else:
        from prediction_model.model.ns_transformer.trainer import predict as pred_func
    pred, label, x = pred_func(data, startdate, field)
    return {"dates": x, "truth": label, "predicts": pred}


