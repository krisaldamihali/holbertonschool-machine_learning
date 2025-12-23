#!/usr/bin/env python3
"""A script that performs K-means clustering."""
import numpy as np


def initialize(X, k):
    """
    A function that initializes cluster centroids for K-means.
    """

    _, d = X.shape

    centroids = np.random.uniform(low=np.min(
        X, axis=0), high=np.max(X, axis=0), size=(k, d))

    return centroids


def kmeans(X, k, iterations=1000):
    """
    A function that performs K-means on a dataset.
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(k, int) or k <= 0:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    C = initialize(X, k)

    for _ in range(iterations):
        previous_C = np.copy(C)
        cluster_sum = np.zeros_like(C)
        cluster_count = np.zeros((k, 1))

        distances = np.linalg.norm(X[:, np.newaxis, :] - C, axis=-1)
        labels = np.argmin(distances, axis=1)

        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) == 0:
                C[i] = initialize(X, 1)[0]
            else:
                cluster_sum[i] = np.sum(cluster_points, axis=0)
                cluster_count[i] = cluster_points.shape[0]

        non_empty = cluster_count.flatten() != 0
        C[non_empty] = cluster_sum[non_empty] / cluster_count[non_empty]

        distances = np.linalg.norm(X[:, np.newaxis, :] - C, axis=-1)
        labels = np.argmin(distances, axis=1)

        if np.array_equal(C, previous_C):
            return C, labels

    return C, labels
