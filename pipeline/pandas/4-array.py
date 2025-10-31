import numpy as np
'''
A script that select the last 10 rows of
'High' and 'Close' columns and return them as a numpy array
'''


def array(df):
    '''
    A function that select the last 10 rows of
    'High' and 'Close' columns and return them as a numpy array
    '''
    last_10 = df[['High', 'Close']].tail(10)
    return last_10.to_numpy()
