from flask import Flask, render_template, jsonify, request
from prediction_model.get_datas import get_data
from prediction_model.prediction import *

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
    stock_price, Share.all_data = get_data(Share.stock_code, Share.field, Share.training_start_date, Share.test_end_date)
    return jsonify(stock_price)


@app.route('/api/stock_prediction', methods=['GET'])
def get_stock_prediction():
    model = request.args.get("model")
    stock_prediction = prediction(Share.all_data, Share.test_start_date, model, Share.field)
    # 在这里调用预测股票价格的模型，使用stock_code作为参数
    return jsonify(stock_prediction)


if __name__ == '__main__':
    app.run(debug=True)
