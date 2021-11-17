import numpy as np

class initWeight:
    def __init__(self, i, o, shape):
        self.fin = i # input neuron의 수
        self.fout = o # output neuron의 수
        self.shape = shape # weight의 shape 저장

        self.type = None

    # 통상적으로 빈번하게 사용
    def Xavier(self):
        self.type = "Xavier"
        std = (2/(self.fin + self.fout)) ** 0.25
        return np.random.normal(scale=std, size=self.shape)

    # Activate Function이 ReLU인 경우
    def He(self):
        self.type = "He"
        std = np.sqrt(2/self.fin)
        return np.random.normal(scale=std, size=self.shape)


    # RNN에서 많이 사용
    # CNN은 잘 쓰지 않음, Dense Layer에서도 효과가 있다고 알려짐
    def Orthogonal(self):
        self.type = "Orthogonal"
        flat_shape = (self.shape[0], np.prod(self.shape[1:]))
        W = np.random.normal(0, 1, flat_shape)

        u, _, v = np.linalg.svd(W, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(self.shape)

        return q

    def randn(self):
        self.type = "randn"
        return np.random.normal(0, 1, self.shape)
