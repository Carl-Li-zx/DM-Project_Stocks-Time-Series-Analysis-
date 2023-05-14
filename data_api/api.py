import pandas as pd
from .constant import *


def get_stock_data(code, start_date, end_date, options="Currency=CNY;PriceAdj=B", sava_data=True):
    """
    数据接口部分 在wind终端下载所需股票的数据
    说明：
    输入数据举例：

    :param code: 选中的股票代码 codes = 600030.SH
    :param start_date: 数据开始时间 start_date = '2023-04-15'
    :param end_date: 数据截至时间 end_date = '2023-04-30'
    :param options: 以字符串的形式集成多个参数，具体见代码生成器。如无相关参数设置，可以不给option赋值
    :return:
    """
    from WindPy import w
    assert code in CODE_LIST, f"code <{code}> should in constant code list, see constant.py"

    w.start()  # 读取数据
    alldata = w.wsd(code, SUPPORT_FIELDS, start_date, end_date, options)
    data0 = pd.DataFrame(alldata.Data).T
    datas = data0.rename(columns=pd.Series(SUPPORT_FIELDS), index=pd.Series(alldata.Times))
    # datas中列名是股票的开收高低交易量，行索引是日期

    if sava_data:
        save_path = "../dataset/"
        datas.to_csv(save_path + f"{code}.csv")

    return datas
