#!/usr/bin/env python3
"""
A script that calculates the probability density
function of a multivariate Gaussian distribution.
"""
import numpy as np


def pdf(data_points, mean, covariance):
    """
    A function that calculates the probability density
    function of a multivariate Gaussian distribution.
    """

    if not isinstance(data_points, np.ndarray) or len(data_points.shape) != 2:
        return None

    num_features = data_points.shape[1]

    if not isinstance(mean, np.ndarray) or len(mean.shape) != 1:
        return None
    if mean.shape[0] != num_features:
        return None

    if not isinstance(covariance, np.ndarray) or len(covariance.shape) != 2:
        return None
    if covariance.shape[0] != num_features or \
            covariance.shape[1] != num_features:
        return None

    det_sqrt = np.linalg.det(covariance)**0.5
    inv_covariance = np.linalg.inv(covariance)
    difference = data_points - mean

    exponent = -0.5 * np.einsum('ij,ji->i',
                                difference @ inv_covariance, difference.T)

    normalization_constant = 1 / ((2 * np.math.pi)**(num_features/2)
                                  * det_sqrt)
    probabilities = normalization_constant * np.exp(exponent)
    probabilities = np.maximum(probabilities, 1e-300)

    return probabilities
