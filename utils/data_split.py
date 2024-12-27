#设时间窗口长度为2，利用 (T-N, T-1) 时间段的数据作为训练数据，T 时刻的腐蚀程度作为标签
#生成训练数据-标签对
import numpy as np
def generate_pair(x, y, ts):
    length = len(x)
    start, end = 0, length - ts
    data = []
    label = []
    for i in range(end):
        data.append(x[i: i+ts, :])
        label.append(y[i+ts])
    return np.array(data, dtype=np.float64), np.array(label, dtype=np.float64)