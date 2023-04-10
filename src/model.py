import torch
from torch.nn import LSTM, Dropout, Linear, Sequential
import torch.nn as nn


class LSTM4Price:
    def __init__(self, feature_dim=1, hidden_dim=32):
        self.model = Sequential(
            LSTM(feature_dim, hidden_dim),
            Dropout(0.15),
            nn.ReLU(),
            LSTM(feature_dim, hidden_dim),
            Dropout(0.15),
            nn.ReLU(),
            Linear(hidden_dim, feature_dim)
        )

    def forward(self, x_train, y_train, lr=1e-5, num_epoch=20):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(device)
        loss_func = nn.MSELoss().to(device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        for epoch in range(num_epoch):
            step = 0
            self.model.train()
            for x, y in zip(x_train, y_train):
                step += 1
                optimizer.zero_grad()
                x = x.to(device)
                y = y.to(device)
                pred_y = self.model(x)
                loss = loss_func(pred_y, y)
                loss.backward()
                optimizer.step()
                if (step + 1) % 50 == 0:
                    print(loss)

