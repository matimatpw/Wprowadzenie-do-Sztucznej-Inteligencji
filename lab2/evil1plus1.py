import random
import numpy as np
from matplotlib import pyplot as plt
from cec2017_test import F1, f1, f9
import copy

#odpalic pare razy algorytm na tych samych danych poczatkowych
class optim_result:
    def __init__(self) -> None:
        self.my_iteration_history = []
        self.my_f_value_history = []

        self.my_last_x = None

        self.exit_info = "\nAlgorithm reached iteration limit"


    @property
    def iteration_history(self) -> None:
        return self.my_iteration_history

    @property
    def func_value_history(self) -> None:
        return self.my_f_value_history

    @property
    def last_x(self) -> None:
        return self.my_last_x

    def set_last_x(self,last_x:np.ndarray) -> None:
        self.my_last_x = last_x

    def add_iteration(self,iter:int) -> None:
        self.my_iteration_history.append(iter)

    def add_func_value(self,value:float) -> None:
        self.my_f_value_history.append(value)

    def check_if_satisfied(self,start_check_val=500,elements=100)-> bool:
        last_n_elements = self.func_value_history[(-elements):]
        if len(set(last_n_elements)) == 1 and len(self.func_value_history) > start_check_val :
            self.exit_info = f"\nValue is the same for the last >{elements}< iterations"
            return True
        return False

    def __str__(self) -> str:
        exit_info = self.exit_info
        last_x_info = f"\nLast x -> {self.last_x}"
        min_f_info = f"\nMinimum f(x) -> {self.func_value_history[-1]:.3f}"
        iteration_info = f"\nExited on iteration -> {self.iteration_history[-1]}"
        return f"{exit_info}{last_x_info}{min_f_info}{iteration_info}"

class optim_params:
    def __init__(self, cr:int, ss:float, mi:int,mm:float, md:float, check_val = 0.2, step_up = 1.22, step_down = 0.22, succes=0) -> None:
        self.my_control_rate = cr
        self.my_step_size = ss
        self.my_max_iterations = mi
        self.my_mean = mm
        self.my_deviation = md

        self.my_success = succes
        self.my_check_value = check_val
        self.my_step_up = step_up
        self.my_step_down = step_down

    @property
    def control_rate(self) -> None:
        return self.my_control_rate

    @property
    def success(self) -> None:
        return self.my_success

    @property
    def step_size(self) -> None:
        return self.my_step_size

    @property
    def max_iter(self) -> None:
        return self.my_max_iterations

    @property
    def deviation(self) -> None:
        return self.my_deviation

    @property
    def mean(self) -> None:
        return self.my_mean


    @property
    def check_value(self) -> None:
        return self.my_check_value

    @property
    def step_up(self) -> None:
        return self.my_step_up

    @property
    def step_down(self) -> None:
        return self.my_step_down

    def set_stepsize(self, new_step:float)-> None:
        self.my_step_size = new_step

    def set_success(self, new_success:int)-> None:
        self.my_success = new_success


def stepsize_adaptation(iteration: int, o_p: optim_params) -> optim_params:
    if iteration % o_p.control_rate == 0:
        if float(o_p.success) / float(o_p.control_rate) > o_p.check_value:
            o_p.set_stepsize(o_p.step_size * o_p.step_up)
        if float(o_p.success) / float(o_p.control_rate) < o_p.check_value:
            o_p.set_stepsize(o_p.step_size * o_p.step_down)
        o_p.set_success(0)
    return o_p


def evolution_1p1(f:callable, x0: [float], o_p:optim_params, stepsize_adaptation: callable) -> optim_result:
    x = x0.copy()
    result = optim_result()

    for iteration in range(o_p.max_iter):
        result.add_iteration(iteration)
        result.add_func_value(round(f(x)[0], 5))

        my_y = np.random.normal(0.,1.,(1,10))
        my_y *= o_p.step_size
        y = x + my_y

        if f(y) <= f(x): 
            o_p.set_success(o_p.success +1)
            x = y

        o_p = stepsize_adaptation(iteration, o_p)

        print(f"iteration: {iteration},\tvalue: {f(x)}")

        if result.check_if_satisfied():
            break
    result.set_last_x(x)
    return result


def main() -> None:
    x = np.random.uniform(-100.0, 100.0, (1,10))

    my_optim_params_DEFAULT = optim_params(5, 1.0, 2000,0.0,1.0)

    my_optim_params_1 = copy.copy(my_optim_params_DEFAULT)
    my_optim_params_2 = copy.copy(my_optim_params_DEFAULT)
    my_optim_params_3 =  copy.copy(my_optim_params_DEFAULT)
    my_optim_params_4 =  copy.copy(my_optim_params_DEFAULT)

    result_1 = evolution_1p1(f1,x,my_optim_params_1,stepsize_adaptation)
    result_2 = evolution_1p1(f1,x,my_optim_params_2,stepsize_adaptation)
    result_3 = evolution_1p1(f1,x,my_optim_params_3,stepsize_adaptation)
    result_4 = evolution_1p1(f1,x,my_optim_params_4,stepsize_adaptation)

    plt.scatter(result_1.iteration_history, result_1.func_value_history, label="First_try", s=10)
    plt.scatter(result_2.iteration_history, result_2.func_value_history, label="Second_try", s=10)
    plt.scatter(result_3.iteration_history, result_3.func_value_history, label="Third_try", s=10)
    plt.scatter(result_4.iteration_history, result_4.func_value_history, label="Third_try", s=10)

    plt.xlabel('Iterations')
    plt.ylabel('f(x) values')
    plt.title('Function_1 (same paremeters plot)')
    plt.legend()
    
    plt.savefig("Function_1_plots.pdf", format="pdf")


if __name__ =="__main__":
    main()



