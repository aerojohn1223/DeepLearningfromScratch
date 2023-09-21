import numpy as np

#Optimizers

##Stochastic Gradient Descent

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

##Momentum

class Momentum:
    def __init__(self, lr = 0.01, momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]

##AdaGrad

class AdaGrad:
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * (grads[key] / np.sqrt(self.h[key] + 1e-7))


#Initialization

##Xavier Initialization

w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num) #node_num is the number of nodes in the previous layer


##He Initialization

w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num) #node_num is the number of nodes in the previious layer

#Dropout

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio #true -> 1, false -> 0
            return x * self.mask #when multiplied, all false becomes 0
        else:
            return x * (1.0 - self.dropout_ratio) #when test, you multiply (1.0-dropout_ratio) to scale. 

    def backward(self, dout):
        return dout*self.mask #so that the nodes that were masked for forward are also masked here