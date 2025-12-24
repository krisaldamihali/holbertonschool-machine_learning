# Clustering Algorithms

This repository contains Python implementations of various clustering algorithms and techniques, including K-means, Gaussian Mixture Models (GMM), and Agglomerative Clustering.

## Overview

The scripts cover the full pipeline of clustering tasks, from initializing centroids and calculating variance to determining the optimum number of clusters using BIC. It includes both custom implementations from scratch and wrappers using `sklearn`.

## Files

### K-means Clustering
*   `0-initialize.py`: Initializes cluster centroids.
*   `1-kmeans.py`: Performs K-means clustering.
*   `2-variance.py`: Calculates total variance (inertia) for a dataset.
*   `3-optimum.py`: Determines the optimum 'k' value.
*   `10-kmeans.py`: K-means implementation using `sklearn`.

### Gaussian Mixture Models (GMM)
*   `4-initialize.py`: Initializes variables (priors, means, covariances) for GMM.
*   `5-pdf.py`: Calculates the Probability Density Function (PDF).
*   `6-expectation.py`: Performs the Expectation step (E-step).
*   `7-maximization.py`: Performs the Maximization step (M-step).
*   `8-EM.py`: Runs the complete Expectation-Maximization algorithm.
*   `9-BIC.py`: Finds the best 'k' using Bayesian Information Criterion (BIC).
*   `11-gmm.py`: GMM implementation using `sklearn`.

### Hierarchical Clustering
*   `12-agglomerative.py`: Performs agglomerative clustering using Ward linkage and `scipy`.

## Requirements

*   Python 3.x
*   NumPy
*   SciPy
*   scikit-learn
*   Matplotlib

## Author
© Krisalda Mihali
