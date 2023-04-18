import csv
import random
import math
# 准备模拟股票价格数据
stock_prices = [round(random.gauss(0, 1)+10, 2) for _ in range(100)]

# 使用 'stock_prices.csv' 文件名将数据写入 CSV 文件
with open('stock_prices.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    # 写入列名
    csv_writer.writerow(['stock_price'])

    # 写入数据
    for price in stock_prices:
        csv_writer.writerow([price])