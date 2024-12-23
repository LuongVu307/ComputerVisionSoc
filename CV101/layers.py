import numpy as np

class RNN:
    #TODO
    ...

class Conv2D:
    def __init__(self, filters, kernel_size, stride=1, activation=None, initializer=None):
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation
        self.initializer = initializer
        self.trainable = True

    def build(self, channel):
        if not self.initializer:
            self.W = np.random.randn(self.kernel_size, self.kernel_size, channel, self.filter) * 0.01
            self.b = np.zeros(self.filters)

    def forward(self, X):
        self.X = X  # Save input for backpropagation
        m, H_in, W_in, C_in = X.shape
        F_h, F_w, _, C_out = self.W.shape

        # Compute output dimensions for 'valid' padding
        H_out = (H_in - F_h) // self.stride + 1
        W_out = (W_in - F_w) // self.stride + 1

        # Initialize output
        self.output = np.zeros((m, H_out, W_out, C_out))

        # Perform convolution
        for i in range(H_out):
            for j in range(W_out):
                x_slice = X[:, i*self.stride:i*self.stride+F_h, j*self.stride:j*self.stride+F_w, :]
                for k in range(C_out):
                    self.output[:, i, j, k] = np.sum(x_slice * self.W[:, :, :, k], axis=(1, 2, 3)) + self.b[k]

        return self.output
    
    def backward(self, dout):
        m, H_in, W_in, C_in = self.X.shape
        F_h, F_w, _, C_out = self.W.shape

        # Initialize gradients
        dX = np.zeros_like(self.X)
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)

        # Compute gradients
        for i in range(dout.shape[1]):  # H_out
            for j in range(dout.shape[2]):  # W_out
                x_slice = self.X[:, i*self.stride:i*self.stride+F_h, j*self.stride:j*self.stride+F_w, :]
                for k in range(C_out):
                    dW[:, :, :, k] += np.sum(x_slice * dout[:, i, j, k][:, None, None, None], axis=0)
                    dX[:, i*self.stride:i*self.stride+F_h, j*self.stride:j*self.stride+F_w, :] += \
                        dout[:, i, j, k][:, None, None, None] * self.W[:, :, :, k]
                db[k] += np.sum(dout[:, i, j, k])

        return dX, dW, db
    

    def update(self, W, b):
        self.W, self.b = W, b        



class MaxPool2D:
    def __init__(self, pool_size=(2, 2), stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.trainable = False

    def forward(self, X):
        self.X = X  # Save the input for backpropagation
        batch_size, height, width, channels = X.shape
        pool_height, pool_width = self.pool_size
        stride = self.stride

        # Calculate the output dimensions
        out_height = (height - pool_height) // stride + 1
        out_width = (width - pool_width) // stride + 1

        # Initialize the output tensor
        output = np.zeros((batch_size, out_height, out_width, channels))

        # Perform max pooling
        for b in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    for c in range(channels):
                        # Define the pooling window
                        h_start = h * stride
                        h_end = h_start + pool_height
                        w_start = w * stride
                        w_end = w_start + pool_width

                        # Apply max pooling
                        window = X[b, h_start:h_end, w_start:w_end, c]
                        output[b, h, w, c] = np.max(window)

        return output

    def backward(self, dout):
        batch_size, height, width, channels = self.X.shape
        pool_height, pool_width = self.pool_size
        stride = self.stride

        # Initialize the gradient for the input
        dX = np.zeros_like(self.X)

        # Backpropagate the gradients
        batch_size, out_height, out_width, channels = dout.shape
        for b in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    for c in range(channels):
                        # Define the pooling window
                        h_start = h * stride
                        h_end = h_start + pool_height
                        w_start = w * stride
                        w_end = w_start + pool_width

                        # Find the max value in the pooling window (from forward pass)
                        window = self.X[b, h_start:h_end, w_start:w_end, c]
                        max_value = np.max(window)

                        # Distribute the gradient to the max value position
                        for i in range(pool_height):
                            for j in range(pool_width):
                                if window[i, j] == max_value:
                                    dX[b, h_start + i, w_start + j, c] += dout[b, h, w, c]

        return dX

class BatchNormalization:
    def __init__(self, epsilon=1e-5, momentum=0.9):
        self.epsilon = epsilon
        self.momentum = momentum
        self.trainable = True

    def build(self, num_features):
        self.num_features = num_features

        # Initialize parameters
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))

        # Running statistics for inference
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))


    def forward(self, X, training=True):
        if not self.built:
            raise ValueError("BatchNormalization layer must be built by calling `build(num_features)` before usage.")

        self.X = X

        if training:
            # Compute batch mean and variance
            self.mean = np.mean(X, axis=0, keepdims=True)
            self.variance = np.var(X, axis=0, keepdims=True)

            # Normalize
            self.X_hat = (X - self.mean) / np.sqrt(self.variance + self.epsilon)

            # Scale and shift
            out = self.gamma * self.X_hat + self.beta

            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.variance
        else:
            # Use running statistics for normalization in inference
            self.X_hat = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            out = self.gamma * self.X_hat + self.beta

        return out

    def backward(self, dout):
        batch_size, num_features = dout.shape

        # Gradients w.r.t. gamma and beta
        dgamma = np.sum(dout * self.X_hat, axis=0, keepdims=True)
        dbeta = np.sum(dout, axis=0, keepdims=True)

        # Gradients w.r.t. normalized input
        dX_hat = dout * self.gamma

        # Gradients w.r.t. variance
        dvar = np.sum(dX_hat * (self.X - self.mean) * -0.5 * np.power(self.variance + self.epsilon, -1.5), axis=0, keepdims=True)

        # Gradients w.r.t. mean
        dmean = np.sum(dX_hat * -1 / np.sqrt(self.variance + self.epsilon), axis=0, keepdims=True) + dvar * np.sum(-2 * (self.X - self.mean), axis=0, keepdims=True) / batch_size

        # Gradients w.r.t. input
        dX = dX_hat / np.sqrt(self.variance + self.epsilon) + dvar * 2 * (self.X - self.mean) / batch_size + dmean / batch_size

        return dX, dgamma, dbeta

class Dropout:
    def __init__(self, rate):
        self.rate = rate  # Fraction of neurons to drop (e.g., 0.2 means 20% dropout)
        self.mask = None  # Mask generated during the forward pass

    def forward(self, X):
        # Create a mask with 1s and 0s, scaled by (1 - rate)
        self.mask = (np.random.rand(*X.shape) > self.rate) / (1 - self.rate)
        return X * self.mask

    def backward(self, dout):
        # Pass gradients only for the retained neurons
        return dout * self.mask

class Flatten:
    def __init__(self):
        self.original_shape = None  # Store original shape for backward pass

    def forward(self, X):
        self.original_shape = X.shape
        # Flatten everything except the first dimension (batch size)
        return X.reshape(X.shape[0], -1)

    def backward(self, dout):
        # Reshape gradient back to the original input shape
        return dout.reshape(self.original_shape)


class Dense:
    def __init__(self, units, activation, use_bias, initializer, regularizer):

        #TODO
        """
        Write code to initialize a Dense layer inputs are 
            - units (int): Number of neurons
            - activation (str): activation function used 
            - use_bias (bool): Whether to use bias
            - initializer (str): initializer used
            - regularizer (Regularizers): regularizers used 
        """

    def build():

        #TODO

        """
        Write code to build the layer given the input shape
        """

    def forward():

        #TODO

        """
        Return the output given the inputs data
        """

    def backward():


        #TODO

        """
        Returns the derivative of Weights and biases  with respect to loss given the output 
        """

    def update():
        ...

