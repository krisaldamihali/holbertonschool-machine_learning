#!/usr/bin/env python3
"""
   A script that calculates the weighted moving average of
   a data set
"""

import numpy as np


def moving_average(data, beta):
    """
        A function that calculates the weighted moving average of
        a data set

        Formul:
        MA = (val1 + val2 +val3 + ... + valN) / N

        :param data: list of data to calculate moving average
        :param beta: weight used for moving average

        :return: list containing the moving averages of data
    """
    m_av = []

    # initialize weight
    w = 0

    for i, d in enumerate(data):
        # update weight average
        w = beta * w + (1 - beta) * d
        # apply bias correction
        w_new = w / (1 - beta ** (i + 1))
        m_av.append(w_new)
    return m_av
