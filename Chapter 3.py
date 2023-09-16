
import numpy as np
import matplotlib.pylab as plt

#sigmoid function

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-10.0, 10.0, 0.2)
y = sigmoid(x)

#softmax function: the wrong way and the right way

## :the wrong way : exponential of numbers are usually very large, so it can cause overflow.
def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x

    return y

## :the right way : overflow is prevented by subtracting the highest number in the x. softmax function's output does not get affected by c.
def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x

    return y


#basic neural network

def init_network(): #weight, bias initialization
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x): #입력 신호를 출력으로 변환하는 처리 과정
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)

# calculating accuracy

def accuracy(network, x, t):
    """
    :param network: network
    :param x: input
    :param t: label
    """

    accuracy_cnt = 0
    for i in range(len(x)):
        y = forward(network, x[i])
        p = np.argmax(y) #get the index of the largest element
        if p == t[i]: #check if predicted label is correct
            accuracy_cnt += 1

    return float(accuracy_cnt) / len(x)

#batch

'''
why batch?
1) most of numerical calculation libraries calculate large arrays efficiently
2) lessen the load caused by frequent data transmission
'''

batch_size = 100

def batch(network, x, t, batch_size=100):
    accuracy_cnt = 0

    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = forward(network, x_batch)
        p = np.argmax(y_batch, axis = 1)
        accuracy_cnt += np.sum(p == t[i:i+batch_size])

    return float(accuracy_cnt) / len(x)