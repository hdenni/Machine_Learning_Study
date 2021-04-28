from CNN.layer import Convolution, maxPooling, Affine, ReLU, Softmax

from tensorflow import keras
import numpy as np


def feedforward(I, answer, layers):
    out = I.copy()

    for layer in layers[:-1]:
        out = layer.forward(out)

    return layers[-1].forward(out, answer)


def backpropagate(layers):
    dout = 1
    for layer in layers:
        dout = layer.backward(dout)

    return


def encode(x_array):
    array = list()
    for x in x_array:
        L = [0 for i in range(10)]
        L[x] = 1

        array.append(L)

    return np.array(array)

def test(y_pred, y_batch):
    return np.unique(y_pred.argmax(axis=1) == y_batch.argmax(axis=1), return_counts=True)

if __name__ == '__main__':
    # 기존 코드는 jupyter notebook에서 개발, 테스트가 진행되어 다른 개발환경에서는 수행되지 않을 수 있음
    # Load Data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(60000, 1, 28, 28) / 255.
    x_test = x_test.reshape(10000, 1, 28, 28) / 255.

    y_train = encode(y_train)
    y_test = encode(y_test)

    train_size = x_train.shape[0]
    batch_size = 1000
    batch_mask = np.random.choice(train_size, batch_size)

    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]

    W1 = np.random.randn(20, 1, 5, 5)
    W2 = np.random.randn(10, 20, 5, 5)
    W3 = np.random.randn(10 * 7 * 7, 10)
    # W4 = np.random.randn(500, 100)
    # W5 = np.random.randn(100, 50)
    # W6 = np.random.randn(50, 10)

    # init Layer
    c1 = Convolution(W1, 0.01, stride=1, pad=2)  # pad = Filter height(or width)/2
    p1 = maxPooling((2, 2), 2)
    c2 = Convolution(W2, 0.01, stride=1, pad=2)
    p2 = maxPooling((2, 2), 2)

    a1 = Affine(W3, 0.01)
    # r1 = ReLU()
    # a2 = Affine(W4, 0.01)
    # r2 = ReLU()
    # a3 = Affine(W5, 0.01)
    # r3 = ReLU()
    # a4 = Affine(W6, 0.01)
    sm = Softmax()

    layers = [c1, p1, c2, p2, a1, sm]
    # layers = [c1, p1, c2, p2, a1, r1, a2, r2, a3, r3, a4, sm]

    # Train
    L = list()
    epochs = 100
    for e in range(epochs):
        feedforward(x_batch, y_batch, layers)
        L.append(sm.loss)
        if L[-1] < 1e-6:
            print("\nLoss is under 0.000001, Early Stopping")
            break
        backpropagate(layers[::-1])
        print(f'\r{e} / {epochs}, loss: {sm.loss}', end='')

    # Check Train
    y_pred = feedforward(x_batch, y_batch, layers)
    test(y_pred, y_batch)

    # Test
    test_size = x_test.shape[0]
    test_batch_size = 1000
    test_batch_mask = np.random.choice(test_size, test_batch_size)

    x_test_batch = x_test[test_batch_mask]
    y_test_batch = y_test[test_batch_mask]

    test_pred = feedforward(x_test_batch, y_test_batch, layers)

    print(test(test_pred, y_test_batch))
