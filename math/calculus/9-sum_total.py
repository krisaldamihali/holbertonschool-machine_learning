#!/usr/bin/env python3
"""A script that calculates the sum"""


def summation_i_squared(n):
    """A function that calculates the sum"""
    if type(n) is not int:
        return None
    return int((n**3)/3 + (n**2)/2 + n/6)
