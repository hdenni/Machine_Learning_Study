from common.layer import Convolution, maxPooling, Affine, ReLU, Softmax
from tensorflow import keras
import numpy as np
from common.common import encode, test
import matplotlib.pyplot as plt
import time


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

def save_file(path, L):
    fig = plt.figure(figsize=(20,10))
    fig.set_facecolor('white')

    plt.plot(range(1,len(L)+1), L, marker='o')
    plt.ylabel('Loss', fontsize=15)
    plt.grid(True)
    plt.savefig(path)

if __name__ == '__main__':
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

    weight_init_std = 0.1
    hidden_size = 100
    output_size = 10

    W1 = np.random.randn(20, 1, 5, 5) * weight_init_std
    W2 = np.random.randn(10, 20, 5, 5) * weight_init_std

    W3 = np.random.randn(490, hidden_size) * weight_init_std
    W4 = np.random.randn(hidden_size, hidden_size) * weight_init_std
    W5 = np.random.randn(hidden_size, output_size) * weight_init_std

    b1 = np.zeros(20)
    b2 = np.zeros(10)

    b3 = np.zeros(hidden_size)
    b4 = np.zeros(hidden_size)
    b5 = np.zeros(output_size)

    # init Layer
    opt = "Adam"

    c1 = Convolution(W1, b1, 0.01, opt=opt, stride=1, pad=2)  # pad = Filter height(or width)/2
    p1 = maxPooling((2, 2), 2)
    c2 = Convolution(W2, b2, 0.01, opt=opt, stride=1, pad=2)
    p2 = maxPooling((2, 2), 2)

    a1 = Affine(W3, b3, 0.01, opt=opt)
    r1 = ReLU()
    # a2 = Affine(W4, b4, 0.01, opt=opt)
    # r2 = ReLU()
    a3 = Affine(W5, b5, 0.01, opt=opt)
    sm = Softmax()  # Lastlayer

    # layers = [c1, p1, c2, p2, a1, r1, a2, r2, a3, sm]
    layers = [c1, p1, c2, p2, a1, r1, a3, sm]
    # layers = [c1, p1, c2, p2, a1, r1, a3, sm]
    # Train
    L = list()
    epochs = 100

    start = time.time()
    for e in range(epochs):
        feedforward(x_batch, y_batch, layers)
        L.append(sm.loss)
        if L[-1] < 1e-6:
            print("\nLoss is under 0.000001, Early Stopping")
            break
        backpropagate(layers[::-1])
        print(f'\r{e} / {epochs}, loss: {sm.loss}', end='')

    end = time.time()

    print("\n Training Time: {} sec".format(end-start))
    print("Loss: {}".format(L[-1]))

    with open("Loss " + opt + ".txt", "w") as f:
        for item in L: f.write("%f\n" % item)

    # Check Train
    y_pred = feedforward(x_batch, y_batch, layers)
    print("Train score: {}".format(test(y_pred, y_batch)))

    # Test
    test_size = x_test.shape[0]
    test_batch_size = 100
    test_batch_mask = np.random.choice(test_size, test_batch_size)

    x_test_batch = x_test[test_batch_mask]
    y_test_batch = y_test[test_batch_mask]

    test_pred = feedforward(x_test_batch, y_test_batch, layers)

    print("Test Score: {}".format(test(test_pred, y_test_batch)))
