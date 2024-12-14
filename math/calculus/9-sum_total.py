#!/usr/bin/env python3
"""Calculus"""


def summation_i_squared(n):
    """Function to calculate the sum of the squares"""

    if not isinstance(n, int) or n <= 0:
        return None

    sum_squares = (n * (n + 1) * (2 * n + 1)) // 6

    return sum_squares
