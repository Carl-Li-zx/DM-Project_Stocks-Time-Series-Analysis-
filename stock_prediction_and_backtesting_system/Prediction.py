import os
import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
from pylab import rcParams

rcParams['figure.figsize'] = 10, 6

import numpy as np

import pandas as pd

'''
  预测模型

'''


def A_prediction(codes, startdate, enddate, datadate):
    # discfile = 'C:/Users/19297/Desktop/作业/研一下/数据挖掘/大作业/data/prices-split-adjusted.csv'

    '''
    说明：
    输入数据举例：

    codes=["600030.SH",'600010.SH','000001.SZ']#选中的股票代码
    startdate='2023-04-15'#测试集开始时间
    enddate='2023-04-30'#测试集结束时间
    datadate='2023-03-01'#训练集起始时间

    '''

    # 数据接口部分 在wind终端下载所需股票的数据

    from WindPy import w
    w.start()
    # 读取数据
    
    alldata = w.wsd(codes, "close", datadate, enddate, "Currency=CNY;PriceAdj=B")

    data0 = pd.DataFrame(alldata.Data).T
    # 以收盘价作为预测目标
    data_close = data0.rename(columns=pd.Series(alldata.Codes), index=pd.Series(alldata.Times))
    # data_close中的每一列都是一只股票的收盘价格，列名是股票代码，行索引是日期

    pred_close = pd.DataFrame()

    for i in range(len(codes)):
        data = data_close[codes[i]]  # 提取单只股票的数据
        # data.drop(['symbol'],axis=1,inplace=True)

        from statsmodels.tsa.arima.model import ARIMA

        train_data, test_data = data[pd.to_datetime(data.index) < pd.to_datetime(startdate)], data[
            pd.to_datetime(data.index) >= pd.to_datetime(startdate)]
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
        print(u'%s 的AIC最小的p值和q值为：%s、%s' % (codes[i], p, q))
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
        pred_close[codes[i]] = predictions
        # error = mean_squared_error(test, predictions)]

    # 设置行索引为日期
    data_example = data_close[codes[1]]
    pred_close = pred_close.set_index(
        data_example[pd.to_datetime(data_example.index) >= pd.to_datetime(startdate)].index)

    # 返回的数据： pred_close 预测出的收盘价， data_close 实际收盘价
    return pred_close, data_close

    # NetValue=pd.DataFrame(NetValue,index=TradingDay)
