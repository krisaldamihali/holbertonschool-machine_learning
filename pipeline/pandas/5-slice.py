#!/usr/bin/env python3
'''
A script that take every 60th row
of the important columns from a DataFrame
'''


def slice(df):
    """
    A function that selects the columns High, Low, Close, and Volume_(BTC),
    then takes every 60th row from the DataFrame.
    """
    selected_columns = df[['High', 'Low', 'Close', 'Volume_(BTC)']]
    sliced_df = selected_columns[::60]
    return sliced_df
