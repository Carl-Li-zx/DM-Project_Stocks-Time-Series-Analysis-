import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings('ignore')


def prediction(data: pd.DataFrame, test_start_date, field):
    data = data[field]
    train_data, test_data = data[pd.to_datetime(data.index.astype(str)) < pd.to_datetime(test_start_date)], data[
        pd.to_datetime(data.index.astype(str)) >= pd.to_datetime(test_start_date)]
    # 定阶
    NHdata = train_data.astype(float)
    pmax = 2
    qmax = 2
    aic_matrix = []  # AIC矩阵
    for p in range(pmax + 1):
        tmp = []
        for q in range(qmax + 1):
            try:  # 存在部分报错，所以用try来跳过报错。
                tmp.append(ARIMA(NHdata, order=(p, 1, q)).fit().aic)
            except:
                tmp.append(None)
        aic_matrix.append(tmp)

    aic_matrix = pd.DataFrame(aic_matrix)  # 从中可以找出最小值

    p, q = aic_matrix.stack().idxmin()  # 先用stack展平，然后用idxmin找出最小值位置。
    print(u'AIC最小的p值和q值为：%s、%s' % (p, q))
    # X = data.values

    # size = len(X) - int(len(X) * 0.9)
    train, test = train_data.values, test_data.values
    predictions = list()
    for t in range(len(test)):
        if t == 0:
            history = train
        else:
            history = np.append(train, test[0:t])
        model = ARIMA(history, order=(p, 1, q))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
    return predictions, test.tolist(), pd.to_datetime(test_data.index.astype(str)).strftime('%Y-%m-%d').tolist()
