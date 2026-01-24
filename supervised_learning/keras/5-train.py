#!/usr/bin/env python3
"""
    A script that trains a model using mini-batch gradient descent
    with optional validation data.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, verbose=True, shuffle=False):
    """
    A function that trains a model using mini-batch gradient descent
    with optional validation data. Returns the training history.
    """
    history = network.fit(x=data,
                          y=labels,
                          epochs=epochs,
                          batch_size=batch_size,
                          validation_data=validation_data,
                          verbose=verbose,
                          shuffle=shuffle)

    return history
