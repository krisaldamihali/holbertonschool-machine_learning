#!/usr/bin/env python3
"""A script that calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """A function that calculates the integral of a polynomial"""
    if type(poly) is not list or len(poly) == 0:
        return None
    if type(C) not in (int, float):
        return None

    for coeff in poly:
        if type(coeff) not in (int, float):
            return None

    result = [C]

    for i in range(len(poly)):
        val = poly[i] / (i + 1)
        if val.is_integer():
            val = int(val)
        result.append(val)

    while len(result) > 1 and result[-1] == 0:
        result.pop()

    return result
