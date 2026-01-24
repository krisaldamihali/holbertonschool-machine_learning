#!/usr/bin/env python3
"""
A script that optimizes a Keras model.
"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    A function that configures the Adam optimizer for a Keras model
    using categorical cross-entropy loss and accuracy metrics,
    with adjustable learning rate and beta parameters.
    """
    Adam_optimizer = K.optimizers.Adam(learning_rate=alpha,
                                       beta_1=beta1,
                                       beta_2=beta2)

    network.compile(optimizer=Adam_optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
