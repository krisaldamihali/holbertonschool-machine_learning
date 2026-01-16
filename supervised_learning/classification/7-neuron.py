#!/usr/bin/env python3
"""
A script with a Neuron class for binary classification
with forward propagation, cost, evaluation, and gradient descent.
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

    def evaluate(self, X, Y):
        """
            A function that evaluates prediction and returns label and cost.
            :param X: ndarray shape(nx,m) contains input data
            :param Y: ndarray shape (1,m) correct labels

            :return: neuron's prediction and cost of the network
        """
        A = self.forward_prop(X)
        return np.where(A >= 0.5, 1, 0), self.cost(Y, A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
            A function that performs one pass of gradient descent.
            :param X: ndarray, shape(nx,m) contains input data
            :param Y: ndarray, shape(1, m) correct labels
            :param A: ndarray, shape(1,m) activated output
            :param alpha: learning rate

            :return: one pass of gradient descent on the neuron
        """
        m = X.shape[1]
        self.__W -= alpha * (1 / m * np.matmul((A - Y), X.T))
        self.__b -= alpha * (1 / m * np.sum(A - Y))

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
            A function that trains the neuron over a number of iterations.
            :param X: ndarray, shape(nx,m) contains input data
            :param Y: ndarray, shape(1, m) correct labels
            :param iterations: number of iterations
            :param alpha: learning rate

            :return: evaluation of the training data after iterations
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        A function that trains the neuron and updates __W, __b, and __A
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if graph:
            import matplotlib.pyplot as plt
            x_points = np.arange(0, iterations + 1, step)
            points = []

        for itr in range(iterations):
            A = self.forward_prop(X)
            if verbose and itr % step == 0:
                cost = self.cost(Y, A)
                print(f"Cost after {itr} iterations: {cost}")
            if graph and itr % step == 0:
                points.append(self.cost(Y, A))
            self.gradient_descent(X, Y, A, alpha)

        if verbose:
            cost = self.cost(Y, A)
            print(f"Cost after {iterations} iterations: {cost}")

        if graph:
            points.append(self.cost(Y, A))
            plt.plot(x_points, points, 'b')
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
