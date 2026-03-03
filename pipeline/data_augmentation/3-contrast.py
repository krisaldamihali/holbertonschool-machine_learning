#!/usr/bin/env python3
"""A script that changes the contrast of an image."""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """A function that changes the contrast of an image."""
    return tf.image.random_contrast(image, lower, upper)
