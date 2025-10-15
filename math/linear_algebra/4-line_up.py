#!/usr/bin/env python3
""" A script that adds two arrays element-wise """


def add_arrays(arr1, arr2):
    if len(arr1) != len(arr2):
        return None

    result = []

    for i in range(len(arr1)):
        sum_elem = arr1[i]+arr2[i]
        result.append(sum_elem)
        
    return result
