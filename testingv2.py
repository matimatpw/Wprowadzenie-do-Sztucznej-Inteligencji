import numpy as np
import matplotlib.pyplot as plt

def objective_function(x):
    return 8 * x**2

def gradient(x):
    return 16 * x

def gradient_descent(learning_rate, num_iterations, x0):
    x = x0
    x_history = []
    f_history = []

    for i in range(num_iterations):
        x = x - learning_rate * gradient(x)
        x_history.append(x)
        f_history.append(objective_function(x))

    return x_history, f_history

learning_rate = 0.1
num_iterations = 50
x0 = 2.0

x_history, f_history = gradient_descent(learning_rate, num_iterations, x0)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(num_iterations), f_history, marker='o')
plt.title('Zbieżność funkcji celu')
plt.xlabel('Numer iteracji')
plt.ylabel('f(x)')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(x_history, f_history, marker='o')
plt.title('Trajektoria spadku gradientu')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()

plt.tight_layout()
plt.show()