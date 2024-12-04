#!/usr/bin/env python3

"""Linear algebra"""


def cat_matrices2D(mat1, mat2, axis=0):

    """Function to cantenate two
    matrices along a specific axis"""

    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2

    elif axis == 1:
        if len(mat1) != len(mat2):
            return None

        result = []

        for i in range(len(mat1)):
            concat_row = mat1[i] + mat2[i]
            result.append(concat_row)

        return result
