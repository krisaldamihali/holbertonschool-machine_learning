#!/usr/bin/env python3
""" A script that concatenates two matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """ A function that concatenates two matrices along a specific axis"""

    if not mat1 or not mat2:
        return None

    if axis == 0:
        for row in mat1:
            if len(row) != len(mat1[0]):
                return None
        for row in mat2:
            if len(row) != len(mat2[0]):
                return None
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2

    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        new_matrix = []
        for i in range(len(mat1)):
            new_matrix.append(mat1[i] + mat2[i])
        return new_matrix

    else:
        return None
