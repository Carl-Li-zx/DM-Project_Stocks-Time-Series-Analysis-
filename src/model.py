import torch
from torch.nn import LSTM, Dropout, Linear, Sequential
import torch.nn as nn


class LSTM4Price(nn.Module):
    def __init__(self, feature_dim=1, hidden_dim=32):
        super().__init__()
        self.lstm1 = LSTM(feature_dim, hidden_dim)
        self.dropout = Dropout(0.15)
        self.lstm2 = LSTM(feature_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear = Linear(hidden_dim, feature_dim)

    def forward(self, x_train):
        o, (hn, cn) = self.lstm1(x_train)
        hn = self.dropout(self.relu(hn))
        cn = self.dropout(self.relu(cn))
        o, (hn, cn) = self.lstm2(x_train, (hn, cn))
        output = self.linear(o)
        return output

