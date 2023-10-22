from autograd import grad
import numpy as np
from functools import partial

#   nr iteracji od wartosci funkcji celu od biezacego x
#   czas od
#
#
#

class optim_result_t:
    def __init__(self, x_table, Beta :float, times: [], iteration_info: int, stop_info: bool) -> None:
        self.learn_rate_info = Beta
        self.time = times
        self.x_history = x_table
        self.time_history = times
        self.iteration_number = iteration_info
        self.stop_info = stop_info

    def __str__(self) -> str:
        body = f"Learn_rate-> {self.learn_rate_info}\nx-> {self.x_history}\n"
        if(self.stop_info):
            return f"{body}Function reached max_iteration_limit and exited!"
        return f"{body}Function >fullfilled< stop_condition and exited on iteration > {self.iteration_number} <"


class optim_params:
    def __init__(self, alfa:float, max_iterations:int, toll: float) -> None:
        self.alfa = alfa # learning_rate
        self.max_iter = max_iterations
        self.toll = toll

def target_func(x):
    return 4*x**2 # 4x^2



def objective_function(x, alpha=1):
    # x is a vector of length 10
    # alpha is a scalar
    # returns a scalar
    n = np.size(x) # -> 10
    x_squared = np.square(x) # squared every element in array
    indexes = np.arange(1, n + 1)
    alpha_values = alpha ** (indexes - 1 / (n - 1))
    values = x_squared * alpha_values
    result = np.sum(values)
    return result




def sphere(x: []):
    return np.sum(x**2)

def solver (func, x0: np.array, params: optim_params) -> optim_result_t: # slownik z tymi parametrami
    best = []
    gradient = grad(func)
    x = x0
    learn_rate = params.alfa
    stop_info = True
    for _ in range(params.max_iter):

        new_x = x - (learn_rate * gradient(x)) # aktualizacja //zmniejszenie wartosci funkcji celu i zblizanie sie do minimum
                                    #tutaj gradient dziala tylko od skalarnej wartosci a nie od wektra
        previous_grad = gradient(x)
        previous_func_val = func(x)

        x = new_x

        #zmienne lokalne przechowujace nowy x i stary x

        # if(new_x==0.0):
        #     print(f"X TO 0 -> iteracja: {_}")
        #     break
        # abs(previous_grad - gradient(new_x)) < params.toll
        if( abs(previous_func_val - func(new_x)) < params.toll):
            iter_info = _
            stop_info = False
            break



    best.append((new_x,f"learn: {learn_rate}" ))

    best.sort()
    print(best)
    my_result = optim_result_t(x,learn_rate,3,iter_info,stop_info)
    return my_result


# --SECTION TO CHOSE VARIABLES-- #
alfas_to_test = np.array([0.1,0.01,0.001])
my_alfa = alfas_to_test[0] # learning_rate
my_max_iter = 1000         # iteration limit
my_toll = 0.0001           # 3 stopper ( Elipse value )


parameters = optim_params(my_alfa, my_max_iter,my_toll)
# print(solver(objective_function,np.array([3.,3.]), parameters))
indexes = np.arange(0, 5,  dtype=int)

print(np.size(indexes))
print(indexes[1])

print(indexes)