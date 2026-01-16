#!/usr/bin/env python3
"""
A script with a Neuron class for binary
classification with forward propagation and cost calculation.
"""
import numpy as np


class Neuron:
    """A class that defines a neuron for binary classification."""

    def __init__(self, nx):
        """A function that initializes weights, bias, and output."""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        A function that returns the weights.

        :return: value for private attribute __W
        """
        return self.__W

    @property
    def b(self):
        """
        A function that returns the bias.

        :return: value for private attribute __b
        """
        return self.__b

    @property
    def A(self):
        """
        A function that returns the activated output.

        :return: value for private attribute __A
        """
        return self.__A

    def forward_prop(self, X):
        """
            A function that calculates forward propagation.
            :param X: ndarray (shape (nx, m)) contains input data

            :return: forward propagation
        """
        self.__A = 1 / (1 + np.exp(-(np.matmul(self.__W, X) + self.__b)))
        return self.__A

    def cost(self, Y, A):
        """
            A function that calculates logistic regression cost.
            :param Y: ndarray shape(1,m) correct labels
            :param A: ndarray shape(1,m) activated output

            :return: the cost
        """
        m = Y.shape[1]
        return -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
