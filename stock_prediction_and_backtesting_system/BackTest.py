import os
import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
from pylab import rcParams

rcParams['figure.figsize'] = 10, 6
from stock_prediction_and_backtesting_system.Prediction import A_prediction
import numpy as np

import pandas as pd

'''
回测系统
'''


def backtest(codes, startdate, enddate, datadate, num, costrate):
    # 根据预测出的收盘价，和实际的前一日的收盘价，计算每日预测收益
    [pred_close, data_close] = A_prediction(codes, startdate, enddate, datadate)

    data_example = data_close[codes[1]]
    close_per_date = data_example.iloc[-len(pred_close) - 1:-1].index
    pred_ret = pd.DataFrame()
    Close = pd.DataFrame()
    Per_Close = pd.DataFrame()
    for j in range(len(codes)):
        close_per = data_close[codes[j]].loc[close_per_date]

        pred_ret[codes[j]] = pd.Series((pred_close[codes[j]].values - close_per.values) / close_per.values)
        close = data_close[codes[j]].loc[pred_close.index]
        Per_Close[codes[j]] = close_per
        Close[codes[j]] = close

    pred_ret.set_index(pred_close.index, inplace=True)

    # h回测部分
    Stkcd = pred_ret.columns
    Bvalue = pd.DataFrame(np.zeros((1, len(Stkcd))), columns=Stkcd)
    TradingDay = pd.Series(pred_ret.index)
    QMatrix = np.zeros((len(TradingDay), len(Stkcd)))
    NetValue = np.zeros(len(TradingDay))
    NetValue[0] = 1

    for i in range(1, len(TradingDay)):
        Today = TradingDay[i]

        TD_stk_buy = pred_ret.loc[Today, :].sort_values(ascending=False)[0:num].index
        Q = QMatrix[i - 1]
        CostRate = costrate
        CostRate_sale = 2 / 10000

        Bvalue[TD_stk_buy] = 1 / num
        QStart = Q
        PT_close0 = close_per[i - 1:i + 1]
        CapitalValue = NetValue[i - 1]
        Pclose = Per_Close.iloc[i, :]
        Tclose = Close.iloc[i, :]

        Bvalue = Bvalue.fillna(0)

        QEnd = np.zeros(len(Stkcd))
        eps = 1e-13
        PL = 0
        # TD_index = list(TD_PdList)

        # 资金分配

        for u in range(len(TD_stk_buy)):
            s = np.where(Stkcd == TD_stk_buy[u])
            Quota = QStart[s]
            PrevClose = Pclose.iloc[s].values
            CL = Tclose.iloc[s].values
            if np.isnan(PrevClose):
                continue
            if np.isnan(CL):
                continue
            TDavilCap = np.array(Bvalue[TD_stk_buy[u]]) * CapitalValue
            ProfitLoss = Quota * (CL - PrevClose)
            QQ = TDavilCap / (CL + eps)
            Cost = abs(QQ - Quota) * CL * CostRate

            ProfitLoss = ProfitLoss - Cost
            PL = PL + ProfitLoss
            QEnd[s] = QQ
        Bvalue.values[:] = 0
        QMatrix[i] = QEnd
        NetValue[i] = PL + NetValue[i - 1]
    NetValue = pd.DataFrame(NetValue, index=TradingDay)
    return NetValue
