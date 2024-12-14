#!/usr/bin/env python3
"""Calculus"""


def poly_integral(poly, C=0):
    """Funciton to calculate the integral of a polynomial"""

    if not poly or not isinstance(poly, list) or not isinstance(C, int):
        return None

    integral = [C]

    for i in range(len(poly)):

        coef = poly[i] / (i + 1)

        if coef.is_integer():
            coef = int(coef)

        integral.append(coef)

    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
