#!/usr/bin/env python3

"""Plotting"""

import numpy as np
import matplotlib.pyplot as plt


def two():

    """Function to plot x ↦ y1 and x ↦ y2 as line graphs"""

    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(x, y1, c='r', label = 'C-14', linestyle="--")
    plt.plot(x, y2, c='g', label = 'Ra-226')
    plt.autoscale(tight=True)
    plt.title('Exponential Decay of Radioactive Elements')
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.legend()
    plt.show()
