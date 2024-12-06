#!/usr/bin/env python3
"""Linear algebra"""

import numpy as np


def np_cat(mat1, mat2, axis=0):

    """Function to concatenate two matrices along a specific axis"""

    return np.concatenate((mat1, mat2), axis)
