#!/usr/bin/env python3
""" A script that concatenates two matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """ A function that concatenates two matrices along a specific axis"""
    if axis == 0:
        return mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        result = []
        for i in range(len(mat1)):
            new_row = mat1[i] + mat2[i]
            result.append(new_row)
        return result
