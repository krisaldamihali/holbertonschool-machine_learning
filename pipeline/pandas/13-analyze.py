#!/usr/bin/env python3
"""A script that computes statistics for a DataFrame"""


def analyze(df):
    """A function that computes statistics for a DataFrame"""
    df = df.drop(columns=['Timestamp']).describe()
    return df
