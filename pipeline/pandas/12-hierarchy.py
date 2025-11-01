#!/usr/bin/env python3
"""A script that combines two tables by timestamp """
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """A function that combines two tables by timestamp """
    df1 = index(df1)
    df2 = index(df2)
    df1_range = df1.loc[1417411980:1417417980]
    df2_range = df2.loc[1417411980:1417417980]
    combined = pd.concat([df2_range, df1_range], keys=['bitstamp', 'coinbase'])
    combined = combined.swaplevel()
    combined = combined.sort_index(level='Timestamp')
    return combined
