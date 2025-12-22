#!/usr/bin/env python3
"""A script that initializes cluster centroids for K-means."""

import numpy as np


def initialize(X, k):
    """
    A function that initializes cluster centroids for K-means.
    """

    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(k, int) or k <= 0):
        return None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    size = (k, X.shape[1])

    centroids = np.random.uniform(low, high, size)

    return centroids
