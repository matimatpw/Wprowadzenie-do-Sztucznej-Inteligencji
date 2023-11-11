import random
import numpy as np

from cec2017_test import F1, f1, f9

#odpalic pare razy algorytm na tych samych danych poczatkowych
class optim_result:
    def __init__(self) -> None:
        self.my_iteration_history = []
        self.my_f_value_history = []

        self.my_last_x = None

        self.exit_info = "\nAlgorithm reached iteration limit"


    @property
    def iteration_history(self):
        return self.my_iteration_history

    @property
    def func_value_history(self):
        return self.my_f_value_history

    @property
    def last_x(self):
        return self.my_last_x

    def set_last_x(self,last_x):
        self.my_last_x = last_x

    def add_iteration(self,iter):
        self.my_iteration_history.append(iter)

    def add_func_value(self,value):
        self.my_f_value_history.append(value)

    def check_if_satisfied(self,start_check_val=1000,elements=5000)-> bool:
        last_200_elements = self.func_value_history[(-elements):]
        # if np.all(last_100_elements) == last_100_elements[0]:
        if len(set(last_200_elements)) == 1 and len(self.func_value_history) > start_check_val :
            print(last_200_elements)
            self.exit_info = f"\nValue is the same for the last >{elements}< iterations"
            return True
        return False

    def __str__(self) -> str:
        exit_info = self.exit_info
        last_x_info = f"\nLast x -> {self.last_x}"
        min_f_info = f"\nMinimum f(x) -> {self.func_value_history[-1]}"
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


#step_size: float, control_rate: int, max_iter

    @property
    def control_rate(self):
        return self.my_control_rate

    @property
    def success(self):
        return self.my_success

    @property
    def step_size(self):
        return self.my_step_size

    @property
    def max_iter(self):
        return self.my_max_iterations

    @property
    def deviation(self):
        return self.my_deviation

    @property
    def mean(self):
        return self.my_mean


    @property
    def check_value(self):
        return self.my_check_value

    @property
    def step_up(self):
        return self.my_step_up

    @property
    def step_down(self):
        return self.my_step_down

    def set_stepsize(self, new_step)-> None:
        self.my_step_size = new_step

    def set_success(self, new_success)-> None:
        self.my_success = new_success




def stepsize_adaptation(iteration: int, o_p: optim_params):
    if iteration % o_p.control_rate == 0: # teraz sprawdzamy czy w przeciagu <a> iteracji
        if float(o_p.success) / float(o_p.control_rate) > o_p.check_value:
            o_p.set_stepsize(o_p.step_size * o_p.step_up)
        if float(o_p.success) / float(o_p.control_rate) < o_p.check_value:
            o_p.set_stepsize(o_p.step_size * o_p.step_down)
        o_p.set_success(0)
    return o_p


def evolution_1p1(f:callable, x0: [float], o_p:optim_params, stepsize_adaptation: callable):
    x = x0.copy()
    result = optim_result()

    for iteration in range(o_p.max_iter):
        result.add_iteration(iteration)
        result.add_func_value(f(x)[0])
# MUTATION (metoda Gaussa)
        my_y = np.random.normal(0.,1.,(1,10))
        my_y *= 1.0
        y = x + my_y

# EVALUATE OFFSPRING
        if f(y) <= f(x): # poprawa potomka
            o_p.set_success(o_p.success +1)
            x = y
#UPDATING mutation_rate based on actual status

        o_p = stepsize_adaptation(iteration, o_p)

        print(f"iteration: {iteration},\tvalue: {f(x)}")

        if result.check_if_satisfied():
            break
    result.set_last_x(x)
    return result







x = np.random.uniform(100.0, 100.0, (1,10))



#TODO czy wykonac program i przy nastepnym uruchomieniu algorytm na korzystac z poprzedniej populacji?
#TODO czy wnioskowac jak algorytm sie szachowuje przy kazdym kolejnym uruchomieniu
#TODO jak z ta 10 wymiarowoscia funckji?


my_optim_params_DEFAULT = optim_params(5, 1.0, 8000,0.0,1.0)

print(evolution_1p1(f1,x,my_optim_params_DEFAULT,stepsize_adaptation))



