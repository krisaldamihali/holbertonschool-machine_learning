#!/usr/bin/env python3
"""
A script that calculates the expectation step in the EM algorithm for a GMM.
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    A function that calculates the expectation step
    in the EM algorithm for a GMM:
    X is a numpy.ndarray of shape (n, d) containing the data set
    pi is a numpy.ndarray of shape (k,) containing the priors for each cluster
    m is a numpy.ndarray of shape (k, d) containing the centroid means for
    each cluster
    S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices
    for each cluster
    You may use at most 1 loop
    Returns: g, l, or None, None on failure
    g is a numpy.ndarray of shape (k, n) containing the posterior probabilities
    for each data point in each cluster
    l is the total log likelihood
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    num_samples, num_features = X.shape

    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not np.allclose(np.sum(pi), 1):
        return None, None

    num_clusters = pi.shape[0]

    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if m.shape[0] != num_clusters or m.shape[1] != num_features:
        return None, None

    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    if S.shape[0] != num_clusters or S.shape[1] != num_features or \
            S.shape[2] != num_features:
        return None, None

    g = np.zeros((num_clusters, num_samples))

    for cluster_idx in range(num_clusters):
        g[cluster_idx] = pi[cluster_idx] * pdf(X, m[cluster_idx],
                                               S[cluster_idx])

    total_density = np.sum(g, axis=0)

    likelihood = np.sum(np.log(total_density))

    g /= total_density

    return g, likelihood
