#!/usr/bin/env python3

"""Linear alegebra"""


def matrix_transpose(matrix0):

    """Funcion to transpose a matrix"""

    new_matrix = []

    for col_index in range(len(matrix0[0])):

        new_row = []
        for row in matrix0:
            new_row.append(row[col_index])

        new_matrix.append(new_row)

    return new_matrix
