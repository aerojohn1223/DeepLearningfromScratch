
import numpy as np

#sum of squares for error(SSE)

def sum_squares_error(y, t):
    return 0.5 * np.sum((y-5)**2)

#cross entropy error(CEE)

def cross_entropy_error(y, t):
    delta = 1e-7 #added to y to prevent we never implement the log of 0(which results negative infinity).
    return -np.sum(t * np.log(y+delta))

#minibatch

def minibatch_data(x, batch_size=10):

    train_size = x.shape[0]
    batch_mask = np.random.choice(train_size, batch_size) #from 0 ~ train_size, batch_size indexes are chosen randomly.
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    return x_batch, t_batch

#cross entropy error for minibatch : OHE ver., number label ver.

## OHE ver.

def cross_entropy_error(y, t):
    if y.ndim == 1: #changing it to 2-D to calculate.
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.og(y + 1e-7)) / batch_size

## number label ver.

def cross_entropy_error(y, t):
    if y.ndim == 1:  # changing it to 2-D to calculate.
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

#gradient

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size): #calculating f(x+h)-f(x-h) / 2*h to avoid the errors caused by the impossibility of diminishing h infinitely to 0.
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad

#gradient descent

def gradient_descent(f, init_x, lr=0.01, step_num = 100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x

#simple learning algorithm(Two layer algorithm)

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        '''
        x = input data
        t = label
        '''

        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_w, self.params['W1'])
        grads['W2'] = numerical_gradient(loss_w, self.params['W2'])
        grads['b1'] = numerical_gradient(loss_w, self.params['b1'])
        grads['b2'] = numerical_gradient(loss_w, self.params['b2'])

        return grads

#mini batch learning

train_loss_list = []
train_acc_list = []
test_acc_list = []

iters_num = 10000
train_size = x_train.shape[0] #x_train is the train set of the data(x)
batch_size = 100
learning_rate = 0.1

iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

for i in range(iters_num):
    #minibatch
    batch_mask = np.random.randn(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #gradient
    grad = network.numerical_gradient(x_batch, t_batch)

    #updating params
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    #accuracy calculating
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

