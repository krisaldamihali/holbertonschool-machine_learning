#!/usr/bin/env python3
"""
A script that defines a NeuralNetwork class for binary classification.
"""


import numpy as np


class NeuralNetwork:
    """
    A class that represents a neural network used for binary classification.
    """

    def __init__(self, nx, nodes):

        """
        A constructor function that initializes a NeuralNetwork object
        using `nx` as the number of input features and `nodes` as the
        number of neurons in the hidden layer.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
