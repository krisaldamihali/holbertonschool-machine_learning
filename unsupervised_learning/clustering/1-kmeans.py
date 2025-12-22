#!/usr/bin/env python3
"""A script that performs K-means clustering."""
import numpy as np


def initialize(X, k):
    """
    A function that initializes cluster centroids for K-means.
    """

    _, d = X.shape
    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)

    centroids = np.random.uniform(
        low=min_values,
        high=max_values,
        size=(k, d)
    )

    return centroids


def kmeans(X, k, iterations=1000):
    """
    A function that performs K-means on a dataset.
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    if not isinstance(k, int) or k <= 0:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    C = initialize(X, k)

    for _ in range(iterations):
        previous_C = C.copy()
        distances = np.linalg.norm(X[:, None, :] - C, axis=2)
        clss = np.argmin(distances, axis=1)
        for i in range(k):
            points_in_cluster = X[clss == i]
            if points_in_cluster.shape[0] == 0:
                C[i] = initialize(X, 1)[0]
            else:
                C[i] = np.mean(points_in_cluster, axis=0)

        if np.array_equal(C, previous_C):
            return C, clss

    return C, clss
