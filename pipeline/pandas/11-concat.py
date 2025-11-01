#!/usr/bin/env python3
""" A script that concatenates DataFrames"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """A function that concatenates DataFrames"""
    df1 = index(df1)
    df2 = index(df2)
    df2 = df2.loc[:1417411920]
    combined = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])

    return combined
