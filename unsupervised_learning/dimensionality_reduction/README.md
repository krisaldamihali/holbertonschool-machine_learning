# Principal Component Analysis (PCA)

This repository implements Principal Component Analysis (PCA) in Python using NumPy and Singular Value Decomposition (SVD).

## Files

### `0-pca.py`
Calculates the weight matrix required to maintain a specific fraction of the dataset's variance.
- **Function:** `pca(X, var=0.95)`
- **Returns:** Weight matrix `W`

### `1-pca.py`
Performs dimensionality reduction to a fixed number of dimensions.
- **Function:** `pca(X, ndim)`
- **Returns:** Transformed dataset `T`

## Requirements

- Python 3.x
- NumPy (`pip install numpy`)

## Author
© Krisalda Mihali
