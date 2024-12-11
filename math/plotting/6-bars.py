#!/usr/bin/env python3

"""Plotting"""

import numpy as np
import matplotlib.pyplot as plt


def bars():

    """Function to plot a stascked bar graph"""

    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    people = ['Farrah', 'Fred', 'Felicia']
    fruits = ['Apples', 'Bananas', 'Oranges', 'Peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

    x = np.arange(len(people))
    bar_width = 0.5

    bottom = np.zeros(len(people))

    # Iterar sobre cada fila de la matriz (tipos de frutas)
    for i in range(fruit.shape[0]):
        plt.bar(
            x,
            fruit[i],
            width=bar_width,
            color=colors[i],
            label=fruits[i],
            bottom=bottom
        )
        bottom += fruit[i]  # Actualizar base para el siguiente nivel

    plt.xticks(x, people)
    plt.yticks(np.arange(0, 81, 10))
    plt.ylim(0, 80)
    plt.ylabel("Quantity of Fruit")
    plt.title("Number of Fruit per Person")
    plt.legend()
    plt.show()
