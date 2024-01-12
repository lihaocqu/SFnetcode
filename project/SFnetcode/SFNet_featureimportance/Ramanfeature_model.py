import torch
import torch.nn as nn
import torch.nn.functional as F

# 基于RNN的模型
class SERSModel(nn.Module):

    def __init__(self):
        super(SERSModel, self).__init__()

        # 初始卷积层和最大池化层
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 创建3个连续的Conv block和Identity block
        self.layer1 = self._make_layer(16, 16, stride=1)
        self.layer2 = self._make_layer(16, 32, stride=2)
        self.layer3 = self._make_layer(32, 64, stride=2)

        # LSTM层
        self.lstm1 = nn.LSTM(input_size=1400, hidden_size=1400, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=2800, hidden_size=300, num_layers=1, batch_first=True, bidirectional=True)

        # 输出层
        self.fc = nn.Linear(in_features=600, out_features=2)
        self.softmax = nn.Softmax(dim=1)

    def _make_layer(self, in_channels, out_channels, stride):
        layers = []
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out, _ = self.lstm1(out)
        out, _ = self.lstm2(out)
        
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out