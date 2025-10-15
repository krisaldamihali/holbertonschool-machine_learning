#!/usr/bin/env python3
""" A script that adds two matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """A function that adds two matrices element-wise"""
    if (len(mat1) != len(mat2)):
        return None

    result = []

    for i in range(len(mat1)):
        row1 = mat1[i]
        row2 = mat2[i]

        if len(row1) != len(row2):
            return None

        new_row = []
        for j in range(len(row1)):
            new_row.append(row1[j]+row2[j])

        result.append(new_row)

    return result
