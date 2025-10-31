#!/usr/bin/env python3
'''
A script that removes any entries where Close has NaN values
and returns DataFrame
'''


def prune(df):
    '''
    A function that removes any entries where Close has NaN values
    and returns DataFrame
    '''
    cleaned_df = df[df['Close'].notna()]
    return cleaned_df
