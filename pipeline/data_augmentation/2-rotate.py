#!/usr/bin/env python3
"""A script that rotates an image 90 degrees counter-clockwise."""
import tensorflow as tf


def rotate_image(image):
    """A function that rotates an image 90 degrees counter-clockwise."""
    return tf.image.rot90(image)
