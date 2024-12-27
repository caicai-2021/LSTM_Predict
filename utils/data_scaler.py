#将数据进行归一化处理
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# 选择第一列为索引列
dataset_default = pd.read_excel('L:/data/2024/10code/LSTM_Predict/data/processed/交互作用分析.xlsx', index_col=0)

class Datascaler:
    def __init__(self):
        pass
    def scaler(self,data:np.array=None):
        '''
        data:输入np参数
        得出标准化后的data和label
        对于时序模型来说，data是包括标签的，同时排除第一列时间索引列，只与序列相关
        对于缺失的数据需要补充相应的算法
        '''
        # 三元运算符判断是否为值
        values = (data.values if data is not None else dataset_default.values).astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        
        data_scaler = np.around(scaled,decimals=3)
        label =np.around(np.array(scaled[:, -1]),decimals=3) 
        print(data_scaler.shape,label.shape)
        print(data_scaler[:5],label[:5])
        return data_scaler,label
    
