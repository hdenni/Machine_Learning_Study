from common.layer import Recurrent, Affine, LSTM
from common.weight import initWeight

from sklearn.metrics import r2_score, accuracy_score

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def split_target(df, start, end, history_size, target_size):
    data = list()
    labels = list()
    for i in range(start, end - history_size):
        indices = range(i, i + history_size)
        indices_label = range(i + history_size, i + history_size + target_size)
        data.append(df[indices])
        labels.append(df[indices_label])

    return np.array(data, dtype=np.float64), np.array(labels, dtype=np.float64)

def norm_data(X, Y):
    new_X = (X - X.min(axis=0)) / np.maximum(X.max(axis=0) - X.min(axis=0), 1e-5)

    overall_mean_Y = np.mean(Y)
    overall_std_Y = np.std(Y)
    new_Y = (Y - overall_mean_Y)/overall_std_Y

    return new_X, new_Y

def draw_pred(y, y_pred, i):
    plt.figure(figsize=(8, 5))
    plt.grid(True)
    plt.xticks(range(len(y[i])))
    plt.plot(range(len(y[i])), y[i], color='b')
    plt.plot(range(len(y_pred[i])), y_pred[i], color='r')

    plt.savefig("Result/y_"+str(i)+".png")

def loss(dy, n):
    return (np.sum(dy, axis=0)[0] ** 2) / n

def save_loss(L):
    plt.figure(figsize=(20, 10))
    plt.plot(L, marker='o')
    plt.grid(True)
    plt.savefig('loss.png')


class SimpleRNN:
    def __init__(self, X, Y, H):
        # input data, label data
        self.X = X
        self.Y = Y

        # self.n = input_shape[1]
        # self.history = input_shape[0]
        # self.target = target
        self.n = X.shape[1]  # 데이터(표본) 수
        self.history = X.shape[0] # train할 데이터의 history size
        self.target = Y.shape[1]  # 최종 예측할 y의 target size

        self.H = H              # hidden layer 뉴런 수
        self.neurons = list()    # neuron (마지막 neuron의 계산은 affine layer와 동일한 방식으로 진행)

    ## Parameter
    # D : input data 차원 수
    # opt : optimizer
    # lr : learning rate
    def set_neuron(self, D, opt, lr):
        neurons = list()

        for i in range(self.history):
            init_Wh = initWeight(self.H, self.H, (self.H, self.H))
            init_Wx = initWeight(D, self.H, (D, self.H))

            # weight 초기화 설정
            Wh = init_Wh.Orthogonal()
            Wx = init_Wx.Orthogonal()

            b = np.zeros(self.H)

            # layer 쌓기
            r = Recurrent(Wx, Wh, b, opt, lr)
            neurons.append(r)

        init_Wy = initWeight(self.H, self.target, (self.H, self.target))
        Wy = init_Wy.Xavier()
        by = np.zeros(self.target)

        a = Affine(Wy, by, lr, "SGD")
        neurons.append(a)

        self.neurons = neurons

    def predict(self, X):
        # initial hidden state는 영향을 끼치지 않도록 0으로 설정 (*)
        h_pred = np.zeros((X.shape[1], self.H), dtype=np.float64)
        for i, neuron in enumerate(self.neurons[:-1]):
            h_pred = neuron.forward(X[i], h_pred)

        y_pred = self.neurons[-1].forward(h_pred)
        return y_pred

    def feedforward(self):
        return self.predict(self.X)

    def backpropagate(self, dout):
        neurons = self.neurons[::-1]

        dh_next = neurons[0].backward(dout)

        for neuron in neurons[1:]:
            dx, dh_next = neuron.backward(dh_next)

        # self.h0 = dh_next
        self.neurons = neurons[::-1]

    def train(self, epochs):
        L = list()
        for e in range(epochs):
            print(f"\rEpochs : {e}", end="")
            y_pred = self.feedforward()

            # update
            dy = y_pred - self.Y
            L.append(loss(dy, self.n))

            self.backpropagate(dy)

        y_pred = self.feedforward()
        # L.append(loss(y_pred - self.Y, self.n))
        # save_loss(L)
        return y_pred

class SimpleLSTM:
    def __init__(self, X, Y, H):
        self.X = X
        self.Y = Y

        self.n = X.shape[1]  # 데이터(표본) 수
        self.history = X.shape[0] # train할 데이터의 history size
        self.target = Y.shape[1]  # 최종 예측할 y의 target size

        self.H = H
        self.neurons = list()
        self.loss = list()

    # H : hidden layer neuron 수가 저장된 list
    # D : input data 차원 수
    def set_neuron(self, D, opt, lr):
        neurons = list()

        for i in range(self.history):
            init_Wx = initWeight(D, self.H*4, (D, self.H*4))
            init_Wh = initWeight(self.H, self.H*4, (self.H, self.H*4))

            # weight 초기화 설정
            Wx = init_Wx.Orthogonal()
            Wh = init_Wh.Orthogonal()

            b = np.zeros(self.H*4)
            b[:self.H] = 1 # forget gate의 bias는 1로 초기화

            # layer 쌓기
            l = LSTM(Wx, Wh, b, opt, lr)
            neurons.append(l)

        init_Wy = initWeight(self.H, self.target, (self.H, self.target))
        Wy = init_Wy.Xavier()
        by = np.zeros(self.target)

        a = Affine(Wy, by, lr, "SGD") # 계산 방식때문에 무조건 SGD여야 한다. 변경불가
        neurons.append(a)

        self.neurons = neurons

    def predict(self, X):
        h_pred = np.zeros((X.shape[1], self.H), dtype=np.float64)
        c_pred = np.zeros((X.shape[1], self.H), dtype=np.float64)
        for i, neuron in enumerate(self.neurons[:-1]):
            h_pred, c_pred = neuron.forward(X[i], h_pred, c_pred)

        y_pred = self.neurons[-1].forward(h_pred)
        return y_pred

    def feedforward(self):
        return self.predict(self.X)

    def backpropagate(self, dout):
        neurons = self.neurons[::-1]

        dh_next = neurons[0].backward(dout)
        dc_next = neurons[1].cache[6] * dh_next
        # cache[6] = o

        for neuron in neurons[1:]:
            dx, dh_next, dc_next = neuron.backward(dh_next, dc_next)

        self.neurons = neurons[::-1]

    def train(self, epochs):
        # L = list()
        for e in range(epochs):
            print(f"\rEpochs : {e}", end="")
            y_pred = self.feedforward()

            # update
            dy = y_pred - self.Y
            self.loss.append(loss(dy, self.n))
            # if len(self.loss)>2 and abs(self.loss[-1] - self.loss[-2]) < 1e-5: break

            self.backpropagate(dy)

        y_pred = self.feedforward()
        self.loss.append(loss(y_pred - self.Y, self.n))
        # save_loss(L)

        return y_pred


if __name__ == '__main__':
    # np.random.seed(0)
    '''
    X = np.array([[[0.1], [0.7], [0.4], [2.5], [1.5]], [[0.2], [0.8], [0.5], [2.6], [1.6]],
                  [[0.3], [0.9], [0.6], [2.7], [1.7]], [[0.4], [1.0], [0.7], [2.8], [1.9]]])

    Y = np.array([[0.5], [1.1], [0.8], [2.9], [2.0]])
    print(X.shape, Y.shape)
    # inputs = input("Input Network Type: ")
    inputs = 'LSTM'
    if inputs == 'RNN':
        # object(X, Y, hidden layer neuron 수 저장하는 list)
        object = SimpleRNN(X, Y, [1, 10, 20, 10, 20])
        print("Simple RNN")
    else:
        object = SimpleLSTM(X, Y, [1, 10, 10, 10, 10])
        print("LSTM")

    object.set_layer(X.shape[2], "Adam", 0.001)
    y_pred = object.train(500)

    # 정답과 예측값 비교
    plt.figure()
    plt.xticks([1,2,3,4,5])
    plt.plot(Y.T[0], marker='o', color='b')
    plt.plot(y_pred.T[0], marker='o', color='r')
    plt.savefig("predict.png")
'''

    df = pd.read_csv("data.csv", index_col=0)
    uni_data = df.values.T[0]

    # train data
    train_data_X, train_data_y = split_target(uni_data, 0, 100, 10, 5)
    train_data_X, train_data_y = norm_data(train_data_X, train_data_y)

    X_train = train_data_X[:, np.newaxis]
    X_train = X_train.transpose(2, 0, 1)
    y_train = train_data_y.copy()

    # test data
    test_data_X, test_data_y = split_target(uni_data, 100, 150, 10, 5)
    test_data_X, test_data_y = norm_data(test_data_X, test_data_y)

    X_test = test_data_X[:, np.newaxis]
    X_test = X_test.transpose(2, 0, 1)
    y_test = test_data_y.copy()

    # train
    obj = SimpleRNN(X_train, y_train, 200)
    obj.set_layer(X_train.shape[2], "Adam", 0.001)
    y_pred = obj.train(1000)

    draw_pred(y_train, y_pred, 4)
    print(f"Train R2 Score: {r2_score(y_train, y_pred)}")
    # print(f"Train Accuracy: {accuracy_score(y_pred, y_train)}")

    # test
    y_pred_test = obj.predict(X_test)
    draw_pred(y_test, y_pred_test, 5)
    print(f"Test R2 Score: {r2_score(y_test, y_pred_test)}")
    # print(f"Test Accuracy: {accuracy_score(y_pred_test, y_train)}")
