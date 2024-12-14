#!/usr/bin/env python3
"""Calculus"""


def poly_derivative(poly):
    """Function to calculate the derivative of a polynomial"""

    if not poly or not isinstance(poly, list):
        return None

    if len(poly) == 1:
        return [0]

    derivative = []

    for i in range(1, len(poly)):
        derivative.append(i * poly[i])

    if not derivative:
        return [0]

    return derivative
