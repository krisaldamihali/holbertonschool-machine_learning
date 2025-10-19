#!/usr/bin/env python3
"""A script that concatenates two matrices along a specific axis"""


def cat_matrices(mat1, mat2, axis=0):
    """A function that concatenates two matrices along a specific axis"""
    if axis == 0:
        if type(mat1[0]) is list:
            if len(mat1[0]) != len(mat2[0]):
                return None
        return mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        result = []
        for i in range(len(mat1)):
            result.append(mat1[i] + mat2[i])
        return result
    return None
