#!/usr/bin/env python3
'''
A script that sets the Timestamp column of a DataFrame as its index.
'''


def index(df):
    '''
    A function that sets the Timestamp column of a DataFrame as its index.
    '''
    df = df.set_index('Timestamp')
    return df
