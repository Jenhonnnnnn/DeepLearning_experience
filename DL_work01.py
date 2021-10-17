import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os


sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定

from common.layers import *
from common.functions import *

from collections import OrderedDict


class TwoLayerNet:

    def __init__(self, input_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, output_size)
        self.params['b1'] = np.zeros(output_size)


        # 生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])

        self.layers['SIGMOID'] = Sigmoid()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        # y_show.append(x)
        return x

    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        # y1 = self.lastLayer.forward(y)
        loss = cross_entropy_error(y, t)
        return loss

    def accuracy(self, x, t):
        y = self.predict(x)

        y = [[1 if n[i] >= 0.5 else 0 for i in range(len(n))] for n in y]

        # print(y == t)
        # y = np.argmax(y, axis=1)
        # if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t ) / float(x.shape[0])
        return accuracy


    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        # dout = self.Lastlayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db


        return grads

# 处理数据
def dataprep(filepath):

    csv_data = pd.read_csv(filepath)
    pd_data = pd.DataFrame(data=csv_data)
    raw_data = []
    for i in range(pd_data.shape[0]):
        raw_data.append([])
        for j in range(pd_data.shape[1]):
            raw_data[i].append(pd_data.iloc[i][j])

    pro_data = [[1 if x[i] == 'Yes' else x[i] for i in range(len(x))] for x in raw_data]
    pro_data_1 = [[0 if x[i] == 'No' else x[i] for i in range(len(x))] for x in pro_data]
    pro_data_2 = [[1 if x[i] == 'Male' else x[i] for i in range(len(x))] for x in pro_data_1]
    pro_data_3 = [[0 if x[i] == 'Female' else x[i] for i in range(len(x))] for x in pro_data_2]
    pro_data_4 = [[0 if x[i] == 'Positive' else x[i] for i in range(len(x))] for x in pro_data_3]
    pro_data_5 = [[1 if x[i] == 'Negative' else x[i] for i in range(len(x))] for x in pro_data_4]
    col_max = np.amax(pro_data_5, axis=0)
    pro_data_6 = np.asarray([sublist for sublist in pro_data_5])
    final_data = pro_data_6 / col_max
    x_data = final_data[:, :-1]
    y_data = final_data[:, [-1]]
    return x_data, y_data

# 数据集分为训练集和测试集
def trainTestSplit(trainingSet, trainingLabels, test_size=0.2):
    totalNum = int(len(trainingSet))
    trainIndex = list(range(totalNum))  # 存放训练集的下标
    x_test = []  # 存放测试集输入
    y_test = []  # 存放测试集输出
    x_train = []  # 存放训练集输入
    y_train = []  # 存放训练集输出
    trainNum = int(totalNum * test_size)  # 划分训练集的样本数
    for i in range(trainNum):
        randomIndex = int(np.random.uniform(0, len(trainIndex)))
        x_test.append(trainingSet[trainIndex[randomIndex]])
        y_test.append(trainingLabels[trainIndex[randomIndex]])
        del (trainIndex[randomIndex])  # 删除已经放入测试集的下标
    for i in range(totalNum - trainNum):
        x_train.append(trainingSet[trainIndex[i]])
        y_train.append(trainingLabels[trainIndex[i]])

    return np.array(x_train), np.asarray(x_test), np.asarray(y_train), np.asarray(y_test)



# 读入数据
x_data, y_data = dataprep('diabetes_data_upload.csv')
x_train,  x_test, t_train, t_test = trainTestSplit(x_data, y_data)

network = TwoLayerNet(input_size=16, output_size=1)


iters_num = 1000
train_size = x_train.shape[0]
batch_size = 26
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
# print(iter_per_epoch)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 梯度
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    # 更新
    for key in ('W1', 'b1'):
        network.params[key] -= learning_rate * grad[key]
        # print(network.params[key])

    loss = network.loss(x_batch, t_batch)
    # print(loss)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        # print(train_acc, test_acc)

x = range(0,1000)
x1 = range(0, 63)

plt.figure("准确率与损失函数")

plt.subplot(2,2,1)
plt.title("Train acccurancy")
plt.plot(x1, train_acc_list)
plt.subplot(2,2,2)
plt.title("Test acccurancy")
plt.plot(x1, test_acc_list)

plt.subplot(2,1,2)
plt.title("Loss")
plt.plot(x, train_loss_list)

plt.show()