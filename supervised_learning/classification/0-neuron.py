#!/usr/bin/env python3
"""
A script with a Neuron class for binary classification.
"""

import numpy as np


class Neuron:
    """
    A class that defines a neuron for binary classification.
    """

    def __init__(self, nx):
        """
        A function that initializes weights, bias, and output for the neuron.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
