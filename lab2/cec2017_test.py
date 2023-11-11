# Using only f5:
from cec2017.functions import f1, f9
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def wrapper_for_1d_function(x_1d):
    # Konwertuj tablicę 1D na 2D o wymiarowości (1, D)
    x_2d = x_1d.reshape(1, -1)
    return x_2d


def F1(x):
    return f1(wrapper_for_1d_function(x))


def F9(x):
    return f9(wrapper_for_1d_function(x))


def wrapper_for_2d_function(x_2d):
    # Konwertuj tablicę 2D o wymiarowości (1, D) na 1D
    x_1d = x_2d.reshape(-1)
    return x_1d


def main():
    # Zakres wartości x, y (ustaw odpowiednio dla swojego obszaru)
    x_range = np.linspace(-100, 100, 100)
    y_range = np.linspace(-100, 100, 100)

    # Tworzenie siatki punktów w obszarze
    X, Y = np.meshgrid(x_range, y_range)

    # Obliczanie wartości funkcji F1 dla każdego punktu na siatce
    Z = np.empty(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([X[i, j], Y[i, j]])
            Z[i, j] = F1(x)

    Z2 = np.empty(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([X[i, j], Y[i, j]])
            Z2[i, j] = F9(x)


    # Rysowanie wykresu 3D dla F1
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis")

    # Dodanie kolorowej mapy
    cbar = fig.colorbar(ax.plot_surface(X, Y, Z, cmap="viridis"), ax=ax, pad=0.1)
    cbar.set_label("Wartość funkcji F1")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Wartość funkcji F1")
    ax.set_title("Wykres 3D funkcji F1 z CEC2017")

    # Rysowanie wykresu 3D dla F9
    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z2, cmap="viridis")

    # Dodanie kolorowej mapy
    cbar = fig2.colorbar(ax.plot_surface(X, Y, Z2, cmap="viridis"), ax=ax, pad=0.1)
    cbar.set_label("Wartość funkcji F9")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Wartość funkcji F9")
    ax.set_title("Wykres 3D funkcji F9 z CEC2017")

    plt.show()


if __name__ == "__main__":
    main()
