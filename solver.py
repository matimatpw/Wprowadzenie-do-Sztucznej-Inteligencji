from autograd import grad
import numpy as np


#   nr iteracji od wartosci funkcji celu od biezacego x
#   czas od
#
#
#

class optim_result_t:
    def __init__(self, x_table: [], Beta :float, times: [], iteration_info: int, stop_info: str) -> None:
        self.learn_rate_info = Beta
        self.time = times
        self.x_history = x_table
        self.time_history = times
        self.iteration_history = iteration_info
        self.stop_info = stop_info





def target_func(x):
    return 4*x**2 # 4x^2

def sphere(x: []):
    return np.sum(x**2)

learning_rates = np.array([0.1,0.01,0.001])
def solver (func, x0: [], learn_rate=learning_rates[0], max_iter=1000, toll=0.0001) -> optim_result_t: # slownik z tymi parametrami
    best = []
    gradient = grad(func)
    x = x0

    for _ in range(max_iter):

        new_x = x - (learn_rate * gradient(x)) # aktualizacja //zmniejszenie wartosci funkcji celu i zblizanie sie do minimum

        previous_grad = gradient(x)
        previous_func_val = func(x)

        x = new_x

        #zmienne lokalne przechowujace nowy x i stary x

        if(new_x==0.0):
            print(f"X TO 0 -> iteracja: {_}")
            break

        if(abs(previous_grad - gradient(new_x)) < toll and abs(previous_func_val - func(new_x)) < toll):
            iter_info = _
            stop_info = f"param: {learn_rate} - very close to 0-> {_}  \t"
            print(stop_info)
            break



    best.append((new_x,f"learn: {learn_rate}" ))

    best.sort()
    print(best)
    my_result = optim_result_t(x,learn_rate,3,iter_info,stop_info)
    return my_result


print(solver(target_func,3.))
