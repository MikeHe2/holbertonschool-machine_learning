#!/usr/bin/env python3
import numpy as np

"""Linear algebra"""


def np_cat(mat1, mat2, axis=0):

    """Function to concatenate two matrices along a specific axis"""

    return np.concatenate((mat1, mat2), axis)
