from autograd import grad
import numpy as np
from functools import partial
# from  time import process_time as pc
import time
#   nr iteracji od wartosci funkcji celu od biezacego x
#   czas od
#
#
#


class optim_result:
    def __init__(self, beta :float) -> None:
        self.learn_rate_info = beta # PARAMETR KROKU

        self.iter_history = []
        self.func_value_history = []

        self.final_time = None

        self.iteration_stop = None
        self.stop_info = None

    def add_func_value(self,fx) -> None:
        self.func_value_history.append(fx)

    def add_iter(self, iter) -> None:
        self.iter_history.append(iter)

    @property
    def get_iterations(self):
        return self.iter_history

    @property
    def get_func_values(self):
        return self.func_value_history

    @property
    def get_beta(self):
        return self.learn_rate_info

    def __str__(self) -> str:
        body = f"Learn_rate-> {self.learn_rate_info}\n"
        if(self.stop_info):
            return f"{body}Function reached max_iteration_limit and exited!"
        return f"{body}Function >fullfilled< stop_condition and exited on iteration > {self.iteration_stop} <"


class optim_params:
    def __init__(self, beta:float, max_iterations:int, toll: float) -> None:
        self.beta = beta # learning_rate
        self.max_iter = max_iterations
        self.toll = toll


#########################################################################

def objective_function(x, alpha):

    # x is a vector of length 10
    # alpha is a scalar
    # returns a scalar
    n = np.size(x) # -> 10
    if(n <=1):
        raise Exception("x must be atleast 2-dimension vector")
    indexes = np.arange(1, n + 1)
    alphas = alpha ** ((indexes - 1) / (n - 1))
    values = np.square(x) * alphas
    result = np.sum(values)
    return result

#########################################################################


def solver (func, x0: np.array, params: optim_params) -> optim_result: # slownik z tymi parametrami

    my_result = optim_result(params.beta)


    gradient = grad(func)

    learn_rate = params.beta
    stop_info = True
    iter_info = params.max_iter
    print(f"TO JEST pierwsza F(x) -> {func(x0)}\n")

    for _ in range(params.max_iter):

        previous_func_val = func(x0)
        # print(previous_func_val)

        my_result.add_iter(_)
        my_result.add_func_value(previous_func_val)


        new_x = x0 - (learn_rate * gradient(x0)) # aktualizacja //zmniejszenie wartosci funkcji celu i zblizanie sie do minimum

        x0 = new_x # aktualizacja x

        if(_ == 999):
            pass
            #if condition to check :
        if( abs(previous_func_val - func(new_x)) < params.toll or np.linalg.norm(gradient(new_x)) < params.toll):

            print(f"TO JEST OSTATNI X -> {new_x}\n")
            print(f"TO JEST OSTATNIa F(x) -> {func(new_x)}\n")
            print(f"TO JEST previous F(x) -> {previous_func_val}\n")
            print(f"TO JEST  abs -> {abs(previous_func_val - func(new_x))}\n")
            print(f"LINEARLG NORM -> {np.linalg.norm(gradient(new_x))}\n")

            iter_info = _
            stop_info = False
            break



    my_result.iteration_stop = iter_info
    my_result.stop_info = stop_info

    return my_result

objective_function_alfa = partial(objective_function, alpha=10)

def main():
    # --SECTION TO CHOSE VARIABLES-- #
    betas_to_test = np.array([0.1,0.01,0.001])
    my_beta = betas_to_test[0] # learning_rate
    my_max_iter = 1000         # iteration limit
    my_toll = 0.0001           # stopper ( Elipse value )
    array =      np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]) * 100.


    parameters = optim_params(my_beta, my_max_iter,my_toll)
    output =  solver(objective_function_alfa,array, parameters)
    print(output)


main()