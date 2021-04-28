import numpy as np

# CNN 폴더 안에 생성되는 common.py

def calcOutputDim(x, fx, stride, pad):
    return (x + 2*pad - fx)//stride + 1


def im2col(input_data, FH, FW, stride=1, pad=0):
    # 4D를 2D로 변환(flatten)

    if len(input_data.shape) == 3:
        input_data = input_data[np.newaxis, :]
        # 데이터의 수가 1개여서 3차원인 경우, 4차원으로 늘려줌

    N, C, H, W = input_data.shape
    OH = calcOutputDim(H, FH, stride, pad)
    OW = calcOutputDim(W, FW, stride, pad)

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')  # input data에 padding 추가
    col = np.zeros((N, C, FH, FW, OH, OW))

    for y in range(FH):
        y_max = y + stride * OH
        for x in range(FW):
            x_max = x + stride * OW
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * OH * OW, -1)

    return col


def col2im(col, input_shape, FH, FW, stride=1, pad=0):
    N, C, H, W = input_shape
    OH = (H + 2 * pad - FH) // stride + 1
    OW = (W + 2 * pad - FW) // stride + 1

    col = col.reshape((N, OH, OW, C, FH, FW))
    # 현재 col의 column = FH * FW * C
    # 현재 col의 row = N * OH * OW
    col = col.transpose(0, 3, 4, 5, 1, 2)
    # reshape 후 transpose

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    # padding, stride 적용 (없다면 input_shape와 동일한 shape로 정의)

    for y in range(FH):
        y_max = y + stride * OH
        for x in range(FW):
            x_max = x + stride * OW
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]  # padding 제외하고 안쪽 값들만 return
