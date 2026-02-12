#!/usr/bin/env python3
"""
    A script that standardizes constants
"""

import numpy as np


def normalization_constants(X):
    """
        A function that calculates the normalization constants
    """
    return np.mean(X, axis=0), np.std(X, axis=0)
