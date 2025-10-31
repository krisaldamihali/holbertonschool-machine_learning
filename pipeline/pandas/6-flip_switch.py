#!/usr/bin/env python3
'''
A script that reverses the order of rows in a DataFrame
and then transposes it
'''


def flip_switch(df):
    """
    A function that sorts the DataFrame in reverse chronological order
    and then transposes it.
    """
    df_reversed = df.sort_index(ascending=False)
    df_transposed = df_reversed.T
    return df_transposed
