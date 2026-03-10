#!/usr/bin/env python3
"""A script that builds the DenseNet-121 architecture module"""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def dense_block_relu(X, nb_filters, growth_rate, layers):
    """Build a dense block using explicit ReLU layers."""
    concatenated_layers = [X]
    current_filters = nb_filters
    init = K.initializers.he_normal(seed=0)

    for _ in range(layers):
        X = K.layers.BatchNormalization()(X)
        X = K.layers.ReLU()(X)
        X = K.layers.Conv2D(
            4 * growth_rate, 1, padding='same', kernel_initializer=init
        )(X)

        X = K.layers.BatchNormalization()(X)
        X = K.layers.ReLU()(X)
        X = K.layers.Conv2D(
            growth_rate, 3, padding='same', kernel_initializer=init
        )(X)

        concatenated_layers.append(X)
        X = K.layers.Concatenate()(concatenated_layers)
        concatenated_layers.pop(0)
        concatenated_layers.pop(0)
        concatenated_layers.append(X)

        current_filters += growth_rate

    return X, current_filters


def densenet121(growth_rate=32, compression=1.0):
    """A function that builds the DenseNet-121 architecture

    Args:
        growth_rate: growth rate for the dense blocks
        compression: compression factor for the transition layers

    Returns:
        the keras model
    """
    init = K.initializers.he_normal(seed=0)
    X = K.Input(shape=(224, 224, 3))

    nb_filters = 2 * growth_rate

    x = K.layers.BatchNormalization(axis=3)(X)
    x = K.layers.ReLU()(x)
    x = K.layers.Conv2D(
        nb_filters, (7, 7), strides=(2, 2), padding='same',
        kernel_initializer=init
    )(x)
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x, nb_filters = dense_block_relu(x, nb_filters, growth_rate, 6)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    x, nb_filters = dense_block_relu(x, nb_filters, growth_rate, 12)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    x, nb_filters = dense_block_relu(x, nb_filters, growth_rate, 24)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    x, nb_filters = dense_block_relu(x, nb_filters, growth_rate, 16)

    x = K.layers.AveragePooling2D((7, 7), strides=(1, 1), padding='valid')(x)
    x = K.layers.Dense(1000, activation='softmax', kernel_initializer=init)(x)

    model = K.models.Model(inputs=X, outputs=x)
    return model
