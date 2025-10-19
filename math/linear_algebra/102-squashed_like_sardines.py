#!/usr/bin/env python3
"""A script that concatenates two matrices along a specific axis"""

import numpy as np


def cat_matrices(mat1, mat2, axis=0):
    """A function that concatenates two matrices along a specific axis"""
    try:
        np_mat1 = np.array(mat1)
        np_mat2 = np.array(mat2)

        result = np.concatenate((np_mat1, np_mat2), axis=axis)

        return result.tolist()

    except Exception:
        return None
