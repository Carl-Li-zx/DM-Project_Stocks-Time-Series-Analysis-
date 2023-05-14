from flask import Flask, render_template, jsonify, request
from prediction_model.get_datas import get_data
from prediction_model.prediction import *
from stock_prediction_and_backtesting_system.BackTest import backtest
import pandas as pd

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/stock_price', methods=['GET'])
def get_stock_price():
    Share.stock_code = request.args.get('stock_code')
    Share.field = request.args.get("field")
    Share.training_start_date = request.args.get('training_start_date')
    Share.test_start_date = request.args.get('test_start_date')
    Share.test_end_date = request.args.get('test_end_date')
    # 在这里调用获取股票价格的函数或API，使用stock_code作为参数
    stock_price, Share.all_data = get_data(Share.stock_code, Share.field, Share.training_start_date,
                                           Share.test_end_date)
    return jsonify(stock_price)


@app.route('/api/stock_prediction', methods=['GET'])
def get_stock_prediction():
    model = request.args.get("model")
    stock_prediction = prediction(Share.all_data, Share.test_start_date, model, Share.field)
    # 在这里调用预测股票价格的模型，使用stock_code作为参数
    return jsonify(stock_prediction)


@app.route('/api/backtest_stock_code', methods=['GET'])
def get_stock_codes():
    # Replace this with actual logic to get stock codes
    stock_codes = Share.codes
    return jsonify(stock_codes)


@app.route('/api/backtest_stock_code', methods=['POST'])
def handle_selected_codes():
    selected_codes = request.json.get('backtest-stock-codes')
    training_start = request.json.get('training-start-date')
    test_start_date = request.json.get('test-start-date')
    test_end_date = request.json.get('test-end-date')
    num = int(request.json.get('num'))
    costrate = float(request.json.get('costrate'))
    backtest_results = backtest(selected_codes, test_start_date, test_end_date, training_start, num, costrate)
    x, y = pd.to_datetime(backtest_results.index.astype(str)).strftime(
        '%Y-%m-%d').tolist(), backtest_results.values.tolist()
    return jsonify({'x': x, 'y': y})


if __name__ == '__main__':
    app.run(debug=True)
