#!/usr/bin/env python3
'''
A script that takes every 60th row
of the important columns from a DataFrame
'''


def slice_df(df):
    '''
    A function that takes every 60th row
    of the important columns from a DataFrame
    '''
    df = df[['High', 'Low', 'Close', 'Volume_(BTC)']]
    df = df[::60]
    return df
