import numpy as np

#im2col -> image to columns

def im2col(input_data, filter_h, filter_w, stride = 1,pad=0):
    """
    converting images to 2-D array

    :param input_data: Input data in 4-D(# of images, # of channels, height, width)
    :param filter_h: height of filters
    :param filter_w: width of filters
    :param stride: stride
    :param pad: padding

    :return: col : 2-D array
    """

    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (H + 2*pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad,pad), (pad, pad)], 'constant') #padding the height and width of input with 0
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w)) #

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.Transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1) #after transpose -> N, out_h, out_w, c, filter_h, filter_w
    return col

#col2im -> columns to images

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    converting 2-D arrays to images

    :param col: 2-D array
    :param input_shape: original shape of the image
    :param filter_h: height of filters
    :param filter_w: width of filters
    :param stride: stride
    :param pad: padding

    :return: img : converted images
    """

    N, C, H, W = input_shape
    out_h = (H + (2*pad) - filter_h) // stride + 1
    out_w = (W + (2*pad) - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2) #trying to make 'col' before transpose and reshape were done(from im2col)
                                                                                         #after transpose -> N, C, filter_h, filter_w, out_h, out_w

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]





#Convolutional Layer

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = 1
        self.pad = 0

        self.x = None
        self.col = None
        self.col_W = None

        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1)
        out = np.dot(col, col_W) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2) #makes it 4-D and then changes it to N, C, H, W

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN) #N, H, W, C. and then makes it 2-D

        self.db = np.sum(dout, axis = 0) #derivative of b
        self.dW = np.dot(self.col.T, dout) #derivative of W
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW) #transposes it so that it becomes (FN, _) , then reshapes it to 4-D

        dcol = np.dot(dout, self.col_W.T) #derivatives of image columns
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad) #columns to image

        return dx

#Pooling Layer

class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.argmax = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = (H + (2*self.pad) - self.pool_h) // self.stride + 1
        out_w = (W + (2*self.pad) - self.pool_w) // self.stride + 1

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w) # to place all the data that is in the pooling application area in a row

        arg_max = np.argmax(col, axis = 1)
        out = np.max(col, axis = 1) #finding the biggest data for every row

        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max
        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1) #after transpose -> N, out_h, out_w, C. reverse of transpose(0, 3, 1, 2)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten() #reverse of reshape(N,out_h,out_w,c)
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx

#ConvNet

class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num':30, 'filter_size':5,
                             'pad':0, 'stride':1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'],
                                           self.params['b1'],
                                           conv_param['stride'],
                                           conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h = 2, pool_w = 2, stride = 2)
        self.layers['Affine1'] = Affine(self.params['W2'],
                                        self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'],
                                        self.params['b3'])
        self.last_layer = SofmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db

        return grads





