#!/usr/bin/env python3
'''A script that loads data from a file'''

import pandas as pd


def from_file(filename, delimiter):
    """A function that loads data from a file as a pandas DataFrame"""
    return pd.read_csv(filename, delimiter=delimiter)
