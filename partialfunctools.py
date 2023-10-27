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

        # for _ in range(max_iter):
        #     x = x - (learn_rate * gradient(x)) # aktualizacja kroku

        #     gradient_magnitude = np.abs(gradient(x))

        #     if(gradient_magnitude < stopper):
        #         print(f"param: {learn_rate} - very close to 0->   \t")
        #         break

        for _ in range(max_iter):
            previous_grad = gradient(x)
            previous_func_val = func(x)

            x = x - (learn_rate * gradient(x)) # aktualizacja //zmniejszenie wartosci funkcji celu i zblizanie sie do minimum

            if(x==0.0):
                print(f"X TO 0 -> iteracja: {_}")
                break
            
            if(abs(previous_grad - gradient(x)) < stopper and abs(previous_func_val - func(x)) < stopper):
                print(f"param: {learn_rate} - very close to 0-> {_}  \t")
                break


        best.append((x,f"learn: {learn_rate}" ))

    best.sort()
    print(best)


solver(target_func,3.)
