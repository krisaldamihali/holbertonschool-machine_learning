#!/usr/bin/env python3
"""
    A script that builds a neural network using Keras Sequential API
"""

from tensorflow import keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
        A function that builds a neural network with the Keras library

        :param nx: number of input features to the network
        :param layers: list, number nodes in each layer
        :param activations: list, activation functions for each layer
        :param lambtha: L2 regularization parameter
        :param keep_prob: proba node kept for dropout

        :return: keras model
    """
    model = K.Sequential()

    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(layers[i],
                      activation=activations[i],
                      kernel_regularizer=K.regularizers.L2(lambtha),
                      input_dim=nx))
        else:
            model.add(K.layers.Dense(layers[i],
                      activation=activations[i],
                      kernel_regularizer=K.regularizers.L2(lambtha)))

        if i != len(layers) - 1 and keep_prob is not None:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
