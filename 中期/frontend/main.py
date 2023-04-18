
import gradio as gr
from gradio import Blocks
import random
import io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd



def stock_predict(model_name, num_predictions:int, stock_data:str, upload_file):
    print(type(upload_file))
    # 解析用户输入的股票数据
    stock_data=[0]
    try:
        stock_data = list(map(float, stock_data.strip().split(",")))
    except:
        pass
    try:
        df = pd.read_csv(upload_file.name)
        stock_data = df["stock_price"].tolist()
    except:
        pass
    

    # 调用不同的预测算法得到预测结果
    # if model_name == "ARIMA":
    #     predicted_vals = arima_prediction(data, num_predictions)
    # elif model_name == "LSTM":
    #     predicted_vals = lstm_prediction(data, num_predictions)
    # elif model_name == "Prophet":
    #     predicted_vals = prophet_prediction(data, num_predictions)

    # 随机产生
    predicted_vals=[]
    for i in range(num_predictions):
        predicted_vals.append(random.uniform(min(stock_data)*0.9,max(stock_data)*1.1))
        
    # 将预测结果绘制为图形
    fig = plt.Figure()
    fig.subplots_adjust(top=0.95)
    fig.suptitle("stock prediction")
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(stock_data)), stock_data, label="row data")
    ax.plot(np.arange(len(stock_data), len(stock_data)+len(predicted_vals)),
            predicted_vals, label="prediction")
    ax.legend()
    ax.set_xlabel("time(day)")
    ax.set_ylabel("stock price")
    ax.grid(True)

    # 将图形数据转换成 PNG 格式，以便 Gradio 和 PIL.Image 显示
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)

    # 将 PIL.Image 转换为 NumPy 数组
    np_img = np.array(img)

    return np_img, predicted_vals

def main():
    model_choice = gr.inputs.Dropdown(
        ["ARIMA", "LSTM", "Prophet"],
        label="请选择预测算法：",
        default="ARIMA",
    )

    num_predictions = gr.inputs.Slider(
        label= "请选择预测时间范围：",
        minimum=1,
        maximum=10,
        step=1
    )

    stock_data = gr.inputs.Textbox(
        lines=5,
        label="请输入股票数据（按时间顺序排列）：",
        default="3.5,3.7,4.1,4.0,3.7",
    )
    prediction = gr.inputs.Textbox(
        lines=5,
        label="预测结果",
    )

    gr.Interface(
        fn=stock_predict,
        inputs=[model_choice, num_predictions, stock_data, gr.inputs.File(label="上传csv文件，要求包含stock_price一项")],
        outputs=[gr.outputs.Image("numpy"),prediction],
        title="股票预测模型",
        description="使用不同的预测算法预测股票价格",
        theme="default"
    ).launch()
    # port = random.randint(10000, 20000)
    # app.queue(8).launch(share=False, debug=False, inbrowser=True, server_name="0.0.0.0",server_port=port)
    
if __name__ == "__main__":
    main()