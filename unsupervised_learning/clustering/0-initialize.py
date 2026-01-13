#!/usr/bin/env python3
"""A script that initializes cluster centroids for K-means."""
import numpy as np


def initialize(X, k):
    """
    A function that initializes cluster centroids for K-means.
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    _, d = X.shape

    if not isinstance(k, int) or k <= 0:
        return None

    centroids = np.random.uniform(low=np.min(
        X, axis=0), high=np.max(X, axis=0), size=(k, d))

    return centroids
