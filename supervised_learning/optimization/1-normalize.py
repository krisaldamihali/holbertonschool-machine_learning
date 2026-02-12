#!/usr/bin/env python3
"""
    A script that standardizes a matrix
"""

import numpy as np


def normalize(X, m, s):
    """
        A function that calculates the normalization of a matrix
    """
    # formula of normalisation Z = (X - mean) / std
    Z = (X - m) / s
    return Z
