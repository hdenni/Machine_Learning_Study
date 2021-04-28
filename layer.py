from CNN.common import im2col, col2im, calcOutputDim
import numpy as np

# CNN 폴더 필요

class Convolution:
    def __init__(self, W, lr, stride=1, pad=0):
        # N = 1 / N ** 0.5
        # self.W = np.random.uniform(-N, N, f_shape)
        # self.W = np.random.randn(f_shape[0], f_shape[1], f_shape[2], f_shape[3])
        self.W = W
        # self.b = self.b
        self.lr = lr
        self.stride = stride
        self.pad = pad

        # 중간 데이터（backward 시 사용）
        self.x = None
        self.out = None  # Non-Activate Result(Z)
        self.col = None
        self.col_W = None

        # 가중치와 편향 매개변수의 기울기
        self.dW = None
        # self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape

        # Output layer 크기 계산
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        # 4D to 2D
        # col: input data 2D / col_W: Weight data 2D
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        # calculate (bias는 사용하지 않음)
        out = np.dot(col, col_W)  # + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2) # 기존 input data에 맞추어 reshape&transpose

        self.x = x
        self.col = col
        self.col_W = col_W
        self.out = out

        return np.where(out > 0, out, 0)  # ReLU

    # dout: 이전 레이어에서 전달되어 온 gradient값
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        # gradient 값을 4D에서 2D로 변환
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        # bias
        # self.db = np.sum(dout, axis=0)

        col = np.where(self.col > 0, 1, 0) # inv_relu(x)
        # dW = dout * inv_relu(x)
        self.dW = np.dot(col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        # dx: 다음 레이어로 전달할 gradient값
        # dx = dout * weight
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        # update weight & gradient
        self.W = self.W - self.dW * self.lr
        # self.b = self.b - self.db * self.lr

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
    def __init__(self, W, lr):
        # N = 1 / N ** 0.5
        # self.W = np.random.uniform(-N, N, f_shape)
        # self.W = np.random.randn(f_shape[0], f_shape[1])
        self.W = W
        self.lr = lr

        self.x = None
        self.dW = None
        self.original_shape = None

    def forward(self, x):
        self.original_shape = x.shape

        # Convolution -> Fully Connected 변환을 위해 Flatten
        self.x = x.reshape(x.shape[0], -1)

        out = np.dot(self.x, self.W)

        return out

    def backward(self, dout):
        batch_size = self.x.shape[0]
        dx = np.dot(dout, self.W.T) #/ batch_size
        self.dW = np.dot(self.x.T, dout) #/ batch_size

        self.W = self.W - self.dW * self.lr

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


def softmax(x):
    x = np.subtract(x, np.max(x, axis=1)[:, None])
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1)[:, None]


def cross_entropy_error(y, t):
    # delta = 1e-7
    # return -np.sum(t * np.log(y + delta))
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
