#!/usr/bin/env python3

"""Linear algebra"""


def matrix_shape(matrix):

    """Function that return the shape of a matrix"""

    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
