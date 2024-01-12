import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_dim=1, lstm_dims=[700, 300], fc_dims=[500, 200, 100]):
        super(RNN, self).__init__()

        # LSTM 层
        lstms = []
        in_dim = input_dim
        for out_dim in lstm_dims:
            lstms.append(nn.LSTM(in_dim, out_dim, batch_first=True))
            in_dim = out_dim
        self.lstm = nn.ModuleList(lstms)

        # Dropout 层
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)
        
        # FC 层
        fc_layers = []
        in_dim = lstm_dims[-1]
        for out_dim in fc_dims:
            fc_layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.fc = nn.ModuleList(fc_layers)
            
    def forward(self, x):
        for lstm in self.lstm:
            x, _ = lstm(x)
        x = self.dropout1(x)
        x = self.dropout2(x)
        for fc in self.fc:
            x = fc(x)
        return x