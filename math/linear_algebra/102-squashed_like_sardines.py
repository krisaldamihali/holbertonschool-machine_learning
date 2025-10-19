#!/usr/bin/env python3
"""A script that concatenates two matrices along a specific axis """


def cat_matrices(mat1, mat2, axis=0):
    """A function that concatenates two matrices along a specific axis"""
    if axis == 0:
        if not same_shape(mat1[0], mat2[0]):
            return None
        return mat1 + mat2

    if len(mat1) != len(mat2):
        return None

    new_matrix = []
    for i in range(len(mat1)):
        merged = cat_matrices(mat1[i], mat2[i], axis - 1)
        if merged is None:
            return None
        new_matrix.append(merged)
    return new_matrix


def same_shape(a, b):
    """ Check if two matrices have the same shape (recursively)."""
    if type(a) is list and type(b) is list:
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            if not same_shape(a[i], b[i]):
                return False
        return True
    if type(a) is not list and type(b) is not list:
        return True
    return False
