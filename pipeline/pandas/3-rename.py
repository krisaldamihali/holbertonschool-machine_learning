#!/usr/bin/env python3
'''
A script that renames 'Timestamp' to 'Datetime',
converts it to datetime format, and keeps only 'Datetime'
and 'Close' columns.
'''

import pandas as pd


def rename(df):
    '''A function that renames 'Timestamp' to 'Datetime',
    converts it to datetime format, and keeps only 'Datetime'
    and 'Close' columns '''
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    df = df[['Datetime', 'Close']]
    return df
