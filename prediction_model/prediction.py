from prediction_model.model.ns_transformer.trainer import *
from prediction_model.model.utils import *
import argparse
import os


class Share:
    stock_code = None
    field = None
    shared_data = None
    training_start_date = None
    test_start_date = None
    test_end_date = None
    all_data = None


def train_ns_transformer():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", default=1024, type=int, help="batch size")
    parser.add_argument("-e", "--epoch", default=50, type=int, help="epochs num")
    args = parser.parse_args()
    config = Config()
    for key in dir(args):
        if not key.startswith("_"):
            setattr(config, key, getattr(args, key))

    logger = load_logger(config)
    try:
        np.random.seed(config.random_seed)
        if config.do_train:
            train(config, logger)
    except Exception:
        logger.error("Run Error", exc_info=True)


def prediction(data, startdate, model="arima", field='close'):
    if model.lower() == "arima":
        from prediction_model.model.arima.arima_model import prediction as pred_func
    else:
        from prediction_model.model.ns_transformer.trainer import predict as pred_func
    pred, label, x = pred_func(data, startdate, field)
    return {"dates": x, "truth": label, "predicts": pred}


if __name__ == "__main__":
    train_ns_transformer()
