import numpy as np
import matplotlib.pyplot as plt
import time

def gradient_descent(f, df, x0, learning_rate, num_iterations):
    x = x0
    history = []
    for i in range(num_iterations):
        gradient = df(x)
        x = x - learning_rate * gradient
        history.append(x)
    return x, history

# Przykładowa funkcja celu i jej pochodna
def sample_function(x):
    return x**2

def sample_function_derivative(x):
    return 2 * x

# Parametry
initial_point = 3.0
iterations = 50

# Badanie wpływu wartości współczynnika uczenia
learning_rates = [0.01, 0.1, 0.5, 1.0]
results = []

for learning_rate in learning_rates:
    start_time = time.time()
    minimum, history = gradient_descent(sample_function, sample_function_derivative, initial_point, learning_rate, iterations)
    end_time = time.time()
    execution_time = end_time - start_time
    results.append((history, execution_time))

# Tworzenie wykresu
plt.figure(figsize=(12, 6))

for i, (history, execution_time) in enumerate(results):
    plt.subplot(2, 2, i+1)
    plt.plot(range(iterations), [sample_function(x) for x in history])
    plt.title(f'Learning Rate: {learning_rates[i]}, Time: {execution_time:.4f} sec')
    plt.xlabel('Iteration')
    plt.ylabel('f(x)')

plt.tight_layout()
plt.show()