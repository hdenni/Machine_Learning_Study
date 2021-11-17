from common.common import im2col, col2im, clip_grads
from common import optimizer

import numpy as np

class Recurrent:
    def __init__(self, Wx, Wh, b, opt, lr):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx, dtype=np.float64), np.zeros_like(Wh, dtype=np.float64), np.zeros_like(b, dtype=np.float64)]
        self.cache = None # 역전파 계산시 사용할 중간 데이터

        self.lr = lr

        if opt == 'Momentum':
            self.optimizer = optimizer.Momentum(self.lr)

        elif opt == "AdaGrad":
            self.optimizer = optimizer.AdaGrad(self.lr)

        elif opt == "RMSProp":
            self.optimizer = optimizer.RMSProp(self.lr)

        elif opt == "Adam":
            self.optimizer = optimizer.Adam(self.lr)

        else: # SGD
            self.optimizer = optimizer.SGD(self.lr)

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        # print(h_prev.shape, Wh.shape)
        # print(x.shape, Wx.shape)

        t = np.matmul(h_prev, Wh) + np.matmul(x, Wx) + b
        h_next = np.tanh(t) # activation

        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(self, dt):
        Wx, Wh, b = self.params
        dWx, dWh, db = self.grads
        x, h_prev, h_next = self.cache

        dt = dt * (1-h_next ** 2)

        dx = np.matmul(dt, Wx.T)
        dWx = np.matmul(x.T, dt)

        # 여기 H*H 형태 확인
        dh_prev = np.matmul(dt, Wh.T)
        dWh = np.matmul(h_prev.T, dt)

        db = np.sum(dt, axis=0)

        self.grads = [dWx, dWh, db]
        self.grads = clip_grads(self.grads, 1.0)

        self.params = self.optimizer.update(self.params, self.grads)

        return dx, dh_prev

class LSTM:
    def __init__(self, Wx, Wh, b, opt, lr):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx, dtype=np.float64), np.zeros_like(Wh, dtype=np.float64), np.zeros_like(b, dtype=np.float64)]
        self.cache = None

        self.lr = lr

        if opt == 'Momentum':
            self.optimizer = optimizer.Momentum(self.lr)

        elif opt == 'AdaGrad':
            self.optimizer = optimizer.AdaGrad(self.lr)

        elif opt == 'RMSProp':
            self.optimizer = optimizer.RMSProp(self.lr)

        elif opt == "Adam":
            self.optimizer = optimizer.Adam(self.lr)

        else: # SGD
            self.optimizer = optimizer.SGD(self.lr)

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        H = int(Wx.shape[1]/4)

        # A = np.matmul(x, Wx) + np.matmul(h_prev, Wh) + b
        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b
        # slice
        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]

        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        # print(f.shape, c_prev.shape, g.shape, i.shape)
        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next

    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        # 구해야 할 것
        # dx, dh_prev, dc_prev, di, df, dg, do
        # dWx, dWh, db
        tanh_c_next = np.tanh(c_next)

        ds = dc_next + (dh_next * o) * (1 - tanh_c_next ** 2)
        # dc_next = (o * dh_next) * (1 - tanh_c_next**2)
        dc_prev = ds * f

        do = tanh_c_next * dh_next
        dg = ds * i
        di = ds * g
        df = ds * c_prev

        # f, i, o : sigmoid
        # g : tanh
        df = f * (1 - f) * df
        di = i * (1 - i) * di
        dg = (1 - g ** 2) * dg
        do = o * (1 - o) * do

        dA = np.hstack((df, dg, di, do))

        # print(dA.shape, x.shape, Wh.shape, h_prev.shape)

        # dx = np.matmul(dA, Wx.T)
        # dWx = np.matmul(x.T, dA)
        # dh_prev = np.matmul(dA, Wh.T)
        # dWh = np.matmul(h_prev.T, dA)

        dx = np.dot(dA, Wx.T)
        dWx = np.dot(x.T, dA)
        dh_prev = np.dot(dA, Wh.T)
        dWh = np.dot(h_prev.T, dA)

        db = np.sum(dA, axis=0)

        self.grads = [dWx, dWh, db]
        self.grads = clip_grads(self.grads, 1.0) # 기울기 폭발 threshold 지정 가능

        self.params = self.optimizer.update(self.params, self.grads)

        return dx, dh_prev, dc_prev

class Convolution:
    def __init__(self, W, b, lr, opt="SGD", stride=1, pad=0):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.lr = lr

        if opt == 'Momentum':
            self.optimizer = optimizer.Momentum(self.lr)

        elif opt == 'AdaGrad':
            self.optimizer = optimizer.AdaGrad(self.lr)

        elif opt == 'RMSProp':
            self.optimizer = optimizer.RMSProp(self.lr)

        elif opt == "Adam":
            self.optimizer = optimizer.Adam(self.lr)

        else:  # SGD
            self.optimizer = optimizer.SGD(self.lr)

        self.stride = stride
        self.pad = pad

        # 중간 데이터（backward 시 사용）
        self.x = None
        self.out = None  # Non-Activate Result(Z)
        self.col = None
        self.col_W = None

    def forward(self, x):
        W, b = self.params

        N, C, XH, XW = x.shape
        FN, C, FH, FW = W.shape

        # Output layer 크기 계산
        out_h = 1 + int((XH + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((XW + 2 * self.pad - FW) / self.stride)

        # 4D to 2D
        # col: input data 2D / col_W: Weight data 2D
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = W.reshape(FN, -1).T

        # calculate
        out = np.dot(col, col_W)  + b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2) # 기존 input data에 맞추어 reshape&transpose

        self.x = x
        self.col = col
        self.col_W = col_W
        self.out = out

        return np.where(out > 0, out, 0)  # ReLU

    # dout: 이전 레이어에서 전달되어 온 gradient값
    def backward(self, dout):
        W, b = self.params
        dW, db = self.grads

        FN, C, FH, FW = W.shape
        # gradient 값을 4D에서 2D로 변환
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        # bias
        db = np.sum(dout, axis=0)

        col = np.where(self.col > 0, 1, 0) # inv_relu(x)

        # dW = dout * inv_relu(x)
        dW = np.dot(col.T, dout)
        dW = dW.transpose(1, 0).reshape(FN, C, FH, FW)

        # dx: 다음 레이어로 전달할 gradient값
        # dx = dout * weight
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        # update weight & gradient
        self.params = [W, b]
        self.grads = [dW, db]

        self.params = self.optimizer.update(self.params, self.grads)

        return dx

class maxPooling:
    def __init__(self, p_shape, stride=1, pad=0):
        self.PH = p_shape[0]
        self.PW = p_shape[1]
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None
        self.out = None

    def forward(self, x):
        N, C, H, W = x.shape
        OH = int(1 + (H - self.PH) / self.stride)
        OW = int(1 + (W - self.PW) / self.stride)

        col = im2col(x, self.PH, self.PW, self.stride, self.pad)
        col = col.reshape(-1, self.PH * self.PW)

        # max값에 해당하는 index 저장 (역전파에서 사용)
        arg_max = np.argmax(col, axis=1)

        # max 값 추출
        out = np.max(col, axis=1)
        out = out.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max
        self.out = out

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)  # N, OH, OW, FN

        # numpy.size: numpy 배열의 총 원소 개수
        dmax = np.zeros((dout.size, self.PH * self.PW))

        # numpy.arange: parameter 크기 만큼 linear한 array 생성
        # flatten: 1차원으로 reshape(deep copy)
        # 기존의 reshape는 shallow copy
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (self.PH * self.PW,))  # (1, 1, 2, 2) + (4, ) = (1, 1, 2, 2, 4)=
        # tuple은 값을 수정할 수 없으므로 새로운 tuple을 생성


        N, OH, OW, FN, pool_size = dmax.shape

        dcol = dmax.reshape(N * OH * OW, -1)  # -1: FN * pool_size
        dx = col2im(dcol, self.x.shape, self.PH, self.PW, self.stride, self.pad)

        return dx

class meanPooling:
    def __init__(self, p_shape, stride=1, pad=0):
        self.PH = p_shape[0]
        self.PW = p_shape[1]
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_avg = None
        self.out = None

    def forward(self, x):
        N, C, H, W = x.shape
        OH = int(1 + (H - self.PH) / self.stride)
        OW = int(1 + (W - self.PW) / self.stride)

        col = im2col(x, self.PH, self.PW, self.stride, self.pad)
        col = col.reshape(-1, self.PH * self.PW)
        out = np.mean(col, axis=1)  # 1 Dim average list return

        # reshape
        out = out.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.out = out

        return out

    def backward(self, dout):
        N, C, OH, OW = self.x.shape
        pool_size = self.PH * self.PW

        dout = dout.transpose(0, 2, 3, 1)

        davg = np.zeros((self.col.shape))
        dout = dout.reshape(1, -1).T
        davg = np.repeat(dout / pool_size, repeats=pool_size, axis=1)
        dx = col2im(davg, self.x.shape, self.PH, self.PW, self.stride, self.pad)

        return dx

class Affine:
    def __init__(self, W, b, lr, opt):
        # 에러 확인을 위해 opt="SGD" 제거
        # To Do: 구현 다되면 optimzier 기본 세팅 추가
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.lr = lr

        if opt == "Momentum":
            self.optimizer = optimizer.Momentum(self.lr)

        elif opt == "AdaGrad":
            self.optimizer = optimizer.AdaGrad(self.lr)

        elif opt == "RMSProp":
            self.optimizer = optimizer.RMSProp(self.lr)

        elif opt == "Adam":
            self.optimizer = optimizer.Adam(self.lr)

        else:  # SGD
            self.optimizer = optimizer.SGD(self.lr)


        self.x = None
        self.original_shape = None

    def forward(self, x):
        W, b = self.params

        self.original_shape = x.shape

        # Convolution -> Fully Connected 변환을 위해 Flatten
        self.x = x.reshape(x.shape[0], -1)

        out = np.dot(self.x, W) + b

        return out

    def backward(self, dout):
        W, b = self.params
        dW, db = self.grads

        batch_size = self.x.shape[0]
        dx = np.dot(dout, W.T) #/ batch_size
        dW = np.dot(self.x.T, dout) #/ batch_size
        db = np.sum(dout, axis=0)

        self.params = [W, b]
        self.grads = [dW, db]

        self.grads = clip_grads(self.grads, 100)
        self.params = self.optimizer.update(self.params, self.grads)

        return dx.reshape(self.original_shape)

class Softmax:
    def __init__(self):
        self.loss = None  # 손실
        self.y = None  # softmax의 출력
        self.t = None  # 정답 레이블(원-핫 벡터)

    def forward(self, x, t):
        self.x = x
        self.y = softmax(x)
        self.t = t
        self.loss = cross_entropy_error(self.y, self.t)

        return self.y

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx

class ReLU:
    def __init__(self):
        self.x = None
        self.out = None

    def forward(self, x):
        self.x = x  # input(z)
        self.out = np.maximum(0, x)  # activate result(a)

        return self.out

    def backward(self, dout):
        # X * W + B = A(Z) 라 할 때,
        # gradient = inverse_A(X) + W * 이전 레이어에서 온 gradient
        return np.where(self.x > 0, 1, 0) * dout

class tanh:
    def __init__(self):
        self.x = None
        self.out = None

    def forward(self, x):
        self.x = x
        self.out = np.tanh(x)

        return self.out

    def backward(self, dout):
        return (1-self.out ** 2)*dout

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.nrandn(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask

def softmax(x):
    x = np.subtract(x, np.max(x, axis=1)[:, None])
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1)[:, None]

def cross_entropy_error(y, t):
    # delta = 1e-7
    # return -np.sum(t * np.log(y + delta))
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
