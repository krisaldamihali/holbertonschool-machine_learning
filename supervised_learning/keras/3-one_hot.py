#!/usr/bin/env python3
"""
A script that converts label vectors into one-hot encoded matrices.
"""

import tensorflow.keras as K

def one_hot(labels, classes=None):
    """
        A function that converts a label vector into a one-hot matrix

        :param labels: labels
        :param classes: nbr of classes

        :return: one-hot matrix, shape(labels,classes)
    """
    return K.utils.to_categorical(labels, classes)
