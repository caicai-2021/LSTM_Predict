from keras.models import load_model
from utils.data_scaler import Datascaler
from utils.data_split import generate_pair
from models.model1.Lstm_time import LSTM_time
from sklearn.metrics import r2_score

# 标准化数据
data, label = Datascaler.scaler(Datascaler)
#分理出自变量和标签
var, label_used = generate_pair(data, label, ts=96)

# 划分数据集
train_test_split = int(0.4 * len(label))
train_X = var[0: train_test_split]
train_y = label_used[0: train_test_split]
test_X = var[train_test_split: ]
test_y = label_used[train_test_split: ]

# 创建类实例
model_lstm = LSTM_time(train_X)
# 训练模型
history = model_lstm.train(train_X,train_y,test_X,test_y)
# 保存模型
model_lstm.save('Lstm_time.h5')
# 加载模型
model = load_model('L:/data/2024/10code/Lstm_time.h5')
# 输出绘图数据，训练集和测试集的损失函数
train_loss = history.history['loss']
test_loss = history.history['val_loss']
# 预测数据
y_predict = model.predict(test_X) 
# r2偏差
r_2=r2_score(test_y, y_predict)
print('ok')
# 实时更新预测算法
