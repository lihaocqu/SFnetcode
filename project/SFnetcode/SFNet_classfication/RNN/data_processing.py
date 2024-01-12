import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 数据集类
class SERSDataset(Dataset):

    def __init__(self, data, labels):
        super(SERSDataset, self).__init__()
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

# 数据加载函数
def load_data():
    # 假设我们已经有了一个numpy数组，从中可以获得数据和标签
    # 您需要将此部分替换为实际的数据加载代码
    data = np.load('path_to_your_data.npy') # 替换为实际的数据文件路径
    labels = np.load('path_to_your_labels.npy') # 替换为实际的标签文件路径

    # 数据预处理：移除 960 到 1080 cm-1 的特征
    # 此处需要修改为实际的移除过程
    data = np.delete(data, obj=range(960, 1080), axis=1)
    
    # 数据预处理: 数据归一化
    # 此处需要根据实际情况决定是否使用此步骤
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
    # 将numpy数组转换为torch张量
    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    return data, labels

# 数据集划分函数
def split_dataset(data, labels, train_ratio=0.7, valid_ratio=0.15):
    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=1-train_ratio)
    val_x, test_x, val_y, test_y = train_test_split(test_x, test_y, test_size=valid_ratio/(1-train_ratio))
    
    train_set = SERSDataset(train_x, train_y)
    val_set = SERSDataset(val_x, val_y)
    test_set = SERSDataset(test_x, test_y)

    return train_set, val_set, test_set