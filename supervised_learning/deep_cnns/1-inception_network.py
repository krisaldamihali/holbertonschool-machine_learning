#!/usr/bin/env python3
"""
A script that builds an inception network using Keras model
"""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    A function that builds an inception network using Keras model
    """
    # Weight initializer suitable for layers using ReLU activation
    init = K.initializers.he_normal()

    # Activation function used throughout the network
    activation = K.activations.relu

    # Input tensor for ImageNet-sized images
    img_input = K.Input(shape=(224, 224, 3))

    # Initial convolution layer for low-level feature extraction
    C0 = K.layers.Conv2D(filters=64,
                         kernel_size=(7, 7),
                         strides=(2, 2),
                         padding='same',
                         activation=activation,
                         kernel_initializer=init)(img_input)

    # Spatial downsampling via max pooling
    MP1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                strides=(2, 2),
                                padding='same')(C0)

    # 1x1 convolution for dimensionality reduction
    C2 = K.layers.Conv2D(filters=64,
                         kernel_size=(1, 1),
                         strides=(1, 1),
                         padding='same',
                         activation=activation,
                         kernel_initializer=init)(MP1)

    # 3x3 convolution for higher-level feature extraction
    C3 = K.layers.Conv2D(filters=192,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         padding='same',
                         activation=activation,
                         kernel_initializer=init)(C2)

    # Downsampling prior to Inception modules
    MP4 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                strides=(2, 2),
                                padding='same')(C3)

    # First stack of Inception modules
    I5 = inception_block(MP4, [64, 96, 128, 16, 32, 32])
    I6 = inception_block(I5, [128, 128, 192, 32, 96, 64])

    # Intermediate spatial reduction
    MP7 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                strides=(2, 2),
                                padding='same')(I6)

    # Second stack of Inception modules
    I8 = inception_block(MP7, [192, 96, 208, 16, 48, 64])
    I9 = inception_block(I8, [160, 112, 224, 24, 64, 64])
    I10 = inception_block(I9, [128, 128, 256, 24, 64, 64])
    I11 = inception_block(I10, [112, 144, 288, 32, 64, 64])
    I12 = inception_block(I11, [256, 160, 320, 32, 128, 128])

    # Downsampling before final Inception stage
    MP13 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                 strides=(2, 2),
                                 padding='same')(I12)

    # Final Inception modules
    I14 = inception_block(MP13, [256, 160, 320, 32, 128, 128])
    I15 = inception_block(I14, [384, 192, 384, 48, 128, 128])

    # Global average pooling prior to classification
    AP16 = K.layers.AveragePooling2D(pool_size=(7, 7),
                                     strides=(1, 1),
                                     padding='valid')(I15)

    # Regularization to reduce overfitting
    Dropout17 = K.layers.Dropout(rate=0.4)(AP16)

    # Final classification layer for 1000 ImageNet classes
    output = K.layers.Dense(1000,
                            activation='softmax',
                            kernel_initializer=init)(Dropout17)

    # Assemble the model
    model = K.Model(inputs=img_input, outputs=output)

    return model
