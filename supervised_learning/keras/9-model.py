#!/usr/bin/env python3
"""
    A script that saves and loads model function
"""

import tensorflow.keras as K


def save_model(network, filename):
    """
    A function that saves a complete Keras model to the specified file path.
    """
    network.save(filename)


def load_model(filename):
    """
    A function that loads a complete Keras model from the specified file path.
    Returns the loaded model.
    """
    return K.models.load_model(filename)
