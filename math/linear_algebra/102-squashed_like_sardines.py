#!/usr/bin/env python3
"""A script that concatenates two matrices along a specific axis"""


def cat_matrices(mat1, mat2, axis=0):
    """A function that concatenates two matrices along a specific axis"""

    if axis == 0:
        if type(mat1[0]) == list and len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2

    if len(mat1) != len(mat2):
        return None

    result = []
    for i in range(len(mat1)):
        sub = cat_matrices(mat1[i], mat2[i], axis - 1)
        if sub is None:
            return None
        result.append(sub)
    return result
