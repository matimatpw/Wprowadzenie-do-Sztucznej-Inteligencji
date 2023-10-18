from autograd import grad
import numpy as np


def target_func(x):
    return 4*x**2 # 4x^2
learning_rates = np.array([0.1,0.01,0.001])
def solver (func, x0, params=learning_rates, max_iter=1000, stopper=0.0001):
    best = []
    gradient = grad(func)
    for learn_rate in params:
        x = x0
    # x= x0
        for _ in range(max_iter):
        
            x = x - (learn_rate * gradient(x)) # aktualizacja kroku

            gradient_magnitude = np.abs(gradient(x))
            if(gradient_magnitude < stopper):
                print(f"param: {learn_rate} - very close to 0->   \t")
                break
        best.append((x,f"learn: {learn_rate}" ))

    best.sort()
    print(best)


solver(target_func,3.)
