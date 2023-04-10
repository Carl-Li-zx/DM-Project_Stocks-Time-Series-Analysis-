from src.model import *
import torch


def train(x_train, y_train, lr=1e-5, num_epoch=50):
    model = LSTM4Price()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    loss_func = nn.MSELoss().to(device)
    step = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(num_epoch):

        for x, y in zip(x_train, y_train):
            step += 1
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            pred_y = model(x)
            loss = loss_func(pred_y, y)
            loss.backward()
            optimizer.step()
            if (step + 1) % 50 == 0:
                print(f"epoch:{epoch},step{step},loss{loss}")

    return model


def evaluation(model, prediction_days, model_inputs):
    pass
