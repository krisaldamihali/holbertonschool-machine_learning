#!/usr/bin/env python3
"""A script that adds two matrices"""


def add_matrices(mat1, mat2):
    """A function that adds two matrices"""
    if type(mat1) is list and type(mat2) is list:
        if len(mat1) != len(mat2):
            return None
        result = []
        for i in range(len(mat1)):
            temp = add_matrices(mat1[i], mat2[i])
            if temp is None:
                return None
            result.append(temp)
        return result
    else:
        return mat1 + mat2
