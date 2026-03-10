#!/usr/bin/env python3
"""A script that builds DenseNet-121 Module"""

from tensorflow import keras as K

transition_layer = __import__('6-transition_layer').transition_layer


def dense_block_relu(X, nb_filters, growth_rate, layers):
    """
    Dense block variant that uses explicit ReLU layers for naming parity.
    """
    concatenated_layers = [X]
    current_filters = nb_filters
    init = K.initializers.he_normal(seed=0)

    for _ in range(layers):
        X = K.layers.BatchNormalization()(X)
        X = K.layers.ReLU()(X)
        X = K.layers.Conv2D(
            4 * growth_rate, 1, padding="same", kernel_initializer=init)(X)

        X = K.layers.BatchNormalization()(X)
        X = K.layers.ReLU()(X)
        X = K.layers.Conv2D(
            growth_rate, 3, padding="same", kernel_initializer=init)(X)

        concatenated_layers.append(X)
        X = K.layers.Concatenate()(concatenated_layers)
        concatenated_layers.pop(0)
        concatenated_layers.pop(0)
        concatenated_layers.append(X)

        current_filters += growth_rate

    return X, current_filters


def densenet121(growth_rate=32, compression=1.0):
    """
    A function that builds a Transition Layer.
    """

    X = K.Input(shape=(224, 224, 3))
    nb_filters = 2 * growth_rate
    init = K.initializers.he_normal(seed=0)

    batch_norm = K.layers.BatchNormalization()(X)
    activation = K.layers.ReLU()(batch_norm)
    conv2d = K.layers.Conv2D(nb_filters, 7, strides=2,
                             padding="same",
                             kernel_initializer=init)(activation)
    max_pool = K.layers.MaxPool2D(3, 2, padding="same")(conv2d)

    dense_block_0, nb_filters = dense_block_relu(
        max_pool, nb_filters, growth_rate, 6)
    transition_layer_0, nb_filters = transition_layer(
        dense_block_0, nb_filters, compression)

    dense_block_1, nb_filters = dense_block_relu(
        transition_layer_0, nb_filters, growth_rate, 12)
    transition_layer_1, nb_filters = transition_layer(
        dense_block_1, nb_filters, compression)

    dense_block_2, nb_filters = dense_block_relu(
        transition_layer_1, nb_filters, growth_rate, 24)
    transition_layer_2, nb_filters = transition_layer(
        dense_block_2, nb_filters, compression)

    dense_block_3, nb_filters = dense_block_relu(
        transition_layer_2, nb_filters, growth_rate, 16)

    average_pool = K.layers.AveragePooling2D(
        7, padding="same")(dense_block_3)
    Y = K.layers.Dense(
        1000, "softmax", kernel_initializer=init)(average_pool)

    return K.Model(inputs=X, outputs=Y)
