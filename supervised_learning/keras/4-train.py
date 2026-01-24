#!/usr/bin/env python3
"""
    A script that trains a Keras model using mini-batch gradient descent.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """
    A function that trains a model using mini-batch gradient descent.
    Returns the training history of the model.
    """
    history = network.fit(x=data,
                          y=labels,
                          epochs=epochs,
                          batch_size=batch_size,
                          verbose=verbose,
                          shuffle=shuffle)

    return history
