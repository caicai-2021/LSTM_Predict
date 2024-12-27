from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import adam_v2

class LSTM_time:
    def __init__(self,train_X):
        '''
        对于输入的数据进行数据集的划分
        进行LSTM模型的建模和训练，打印出实时的训练过程
        data:包含自变量+标签的前序时间段
        label:所有标签
        ratio:划分比例
        输出=》
        训练完成的模型
        '''
        # 划分数据集

        self.model = Sequential()
        self.model.add(LSTM(120, input_shape=(train_X.shape[1],train_X.shape[2])))
        self.model.add(Dense(1))
        self.model.summary()
        self.model.compile(loss='mae', optimizer=adam_v2.Adam(learning_rate=0.01))
    
    def train(self,train_X, train_y,test_X,test_y,epochs=20,batch_size=32):
        '''
        训练模型，返回history供展示曲线
        '''
        history = self.model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_X, test_y), verbose=1, shuffle=True)
        return history
    
    def save(self,filepath):
        self.model.save(filepath)
        print('模型已保存至{filepath}')