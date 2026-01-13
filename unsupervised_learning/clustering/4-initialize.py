#!/usr/bin/env python3
"""A script that initializes variables for a Gaussian Mixture Model."""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    A function that initializes the priors, means, and covariances for a GMM.
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    if not isinstance(k, int) or k <= 0:
        return None, None, None

    priors = np.repeat(1.0 / k, k)

    cluster_means, _ = kmeans(X, k)

    num_features = X.shape[1]
    covariances = np.tile(np.identity(num_features), (k, 1, 1))

    return priors, cluster_means, covariances
