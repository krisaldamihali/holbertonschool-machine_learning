#!/usr/bin/env python3
"""A script that changes the brightness of an image."""
import tensorflow as tf


def change_brightness(image, max_delta):
    """A function that changes the brightness of an image."""
    return tf.image.random_brightness(image, max_delta)
