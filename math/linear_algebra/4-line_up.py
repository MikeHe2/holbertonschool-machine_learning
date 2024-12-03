#!/usr/bin/env python3
"""Linear algebra"""


def add_arrays(arr1, arr2):

    """Function that adds two arrays"""

    if len(arr1) != len(arr2):
        return None

    new_matrix = []

    for i in range(len(arr1)):
        new_matrix.append(arr1[i] + arr2[i])

    return new_matrix
