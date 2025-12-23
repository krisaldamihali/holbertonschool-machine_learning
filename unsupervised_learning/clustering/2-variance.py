#!/usr/bin/env python3
"""A script that calculates the total variance for a dataset."""
import numpy as np


def variance(X, C):
    """
    A function that calculates the total variance for a dataset.
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if C.shape[1] != X.shape[1]:
        return None

    distances = np.linalg.norm(X[:, np.newaxis, :] - C, axis=-1)
    closest_clusters = np.argmin(distances, axis=1)
    closest_distances = distances[np.arange(len(X)), closest_clusters]
    total_variance = np.sum(closest_distances ** 2)
    return total_variance
