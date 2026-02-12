#!/usr/bin/env python3
"""
A script that defines function that trains a loaded neural network model
using mini-batch gradient descent
"""


import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    A function that creates mini-batches to be used for training
    a neural network using mini-batch gradient descent.

    Args:
        X (numpy.ndarray): Input data of shape (m, nx).
        Y (numpy.ndarray): Labels of shape (m, ny).
        batch_size (int): Number of data points in a batch.

    Returns:
        list: List of mini-batches containing tuples (X_batch, Y_batch).
    """
    m = X.shape[0]
    mini_batches = []

    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    num_batches = m // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        X_batch = X_shuffled[start_idx:end_idx]
        Y_batch = Y_shuffled[start_idx:end_idx]

        mini_batches.append((X_batch, Y_batch))

    if m % batch_size != 0:
        start_idx = num_batches * batch_size
        X_batch = X_shuffled[start_idx:]
        Y_batch = Y_shuffled[start_idx:]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
