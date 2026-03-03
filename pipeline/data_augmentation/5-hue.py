#!/usr/bin/env python3
"""A script that changes the hue of an image."""
import tensorflow as tf


def change_hue(image, delta):
    """A function that changes the hue of an image."""
    return tf.image.adjust_hue(image, delta)
