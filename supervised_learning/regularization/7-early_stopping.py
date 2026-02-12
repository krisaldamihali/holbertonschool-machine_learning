#!/usr/bin/env python3
"""
    A script that implements early stopping in gradient descent
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
        A function that determines if should stop gradient descent early
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    return count == patience, count
