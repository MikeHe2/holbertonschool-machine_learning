#!/usr/bin/env python3

"""Plotting"""

import numpy as np
import matplotlib.pyplot as plt


def line():

    """Plotting y as a line graph"""

    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(y, c='r')
    plt.autoscale(axis='x', tight=True)
    plt.show()
