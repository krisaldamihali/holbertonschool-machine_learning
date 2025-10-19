#!/usr/bin/env python3
"""A script that slices a matrix along specific axes"""


def np_slice(matrix, axes={}):
    """A function that slices a matrix along specific axes"""
    slices = [slice(None)] * matrix.ndim
    for axis in axes:
        slices[axis] = slice(*axes[axis])
    return matrix[tuple(slices)]
