import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DatasetLoader():
    def __init__(self, filepath, test_size=0.3):
        self.filepath = filepath
        self.test_size = test_size

    def load_data(self):
        # 从 CSV 文件中加载数据
        df = pd.read_csv(self.filepath)
        
        # 将数据标准化
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
        
        # 划分训练集和测试集
        self.train_data, self.test_data = train_test_split(df_scaled, test_size=self.test_size)

    def get_data(self):
        return self.train_data, self.test_data