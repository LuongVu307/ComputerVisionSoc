import numpy as np

class RNN:
    #TODO
    ...

class Conv2D:
    #TODO
    ...

class MaxPooling2D:
    #TODO
    ...

class BatchNormalization:
    #TODO
    ...

class Dropout:
    #TODO
    ...

class Flatten:
    #TODO
    ...





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

    def build(input_shape):

        #TODO

        """
        Write code to build the layer given the input shape
        """

    def compute(inputs):

        #TODO

        """
        Return the output given the inputs data
        """

    def backprop(outputs):


        #TODO

        """
        Returns the derivative of Weights and biases  with respect to loss given the output 
        """


