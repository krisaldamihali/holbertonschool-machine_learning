#!/usr/bin/env python3
""" A script that returns the transpose of a 2D matrix """


def matrix_transpose(matrix):
    """A function that returns the transpose of a 2D matrix"""
    rows = len(matrix)
    cols = len(matrix[0])

    transposed = []
    for i in range(cols):
        new_row = []
        for j in range(rows):
            new_row.append(matrix[j][i])
        transposed.append(new_row)

    return transposed
