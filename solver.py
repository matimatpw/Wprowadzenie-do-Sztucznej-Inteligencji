from autograd import grad
import numpy as np


def target_func(x):
    return 4*x**2 # 4x^2
learning_rates = np.array([0.01,0.001,0.0001])
def solver (func, x0, learn_rates=learning_rates, max_iter=1000, stopper=0.0001):
    best = []
    gradient = grad(func)
    for learn_rate in learn_rates:
        x = x0
        for _ in range(max_iter):
        
            x = x - (0.001 * gradient(x0)) # aktualizacja kroku

            gradient_magnitude = np.abs(gradient(x0))
            if(gradient_magnitude < stopper):
                print(f"very close->   \t")
                break
        best.append((x,f"learn: {learn_rate}" ))


    print(best)


solver(target_func,3.)