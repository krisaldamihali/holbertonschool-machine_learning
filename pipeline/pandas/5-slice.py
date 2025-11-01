#!/usr/bin/env python3
""" A script that slices a pandas DataFrame"""


def slice(df):
    """A function that returns a sliced DataFrame"""
    selected_columns = df[['High', 'Low', 'Close', 'Volume_(BTC)']]
    sliced_df = selected_columns[::60]
    return sliced_df
