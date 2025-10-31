#!/usr/bin/env python3
'''A script that sorts a DataFrame by the High column in descending order'''


def high(df):
    '''
    A function that sorts a DataFrame by the High column in descending order
    '''
    sorted_df = df.sort_values(by='High', ascending=False)
    return sorted_df
