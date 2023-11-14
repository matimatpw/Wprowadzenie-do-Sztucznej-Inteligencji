from autograd import grad
import numpy as np
from functools import partial
from cec2017.functions import f1, f9, f2
import numdifftools as nd


class optim_result:
    def __init__(self, beta: float) -> None:
        self.learn_rate_info = beta  # PARAMETR KROKU

        self.func_value_history = []
        self.iter_history = []

        self.iteration_stop_number = None
        self.stop_information_toggle = None
        # self.final_time = None # ?

    def add_func_value(self, fx: float) -> None:
        self.func_value_history.append(fx)

    def add_iter(self, iter: int) -> None:
        self.iter_history.append(iter)

    def set_iter_stop(self, iter: int) -> None:
        self.iteration_stop_number = iter

    def set_stop_info_toggle(self, info: str) -> None:
        self.stop_information_toggle = info

    @property
    def last_func_value(self) -> float:
        if not self.func_value_history:
            raise Exception("List is empty! No last value!")
        return self.func_value_history[-1]

    @property
    def learn_info(self) -> float:
        return self.learn_rate_info

    @property
    def stop_info_toggle(self) -> bool:
        return self.stop_information_toggle

    @property
    def iter_stop_num(self) -> int:
        return self.iteration_stop_number

    def get_iterations(self) -> []:
        return self.iter_history

    def get_func_values(self) -> []:
        return self.func_value_history

    def __str__(self) -> str:
        body = f"**Learn_rate-> {self.learn_info}\n*Last f(x) value-> {self.get_func_values()[-1]}\n"
        if self.stop_info_toggle:
            return f"{body}*Function reached max_iteration_limit and exited!\n"
        return f"{body}*Function >fullfilled< stop_condition and exited on iteration > {self.iter_stop_num} <\n"


class optim_params:
    def __init__(self, beta: float, max_iterations: int, toll: float) -> None:
        self.my_learn_rate = beta  # learning_rate
        self.my_max_iter = max_iterations
        self.my_toll = toll

    @property
    def learn_rate(self) -> float:
        return self.my_learn_rate

    @property
    def max_iter(self) -> int:
        return self.my_max_iter

    @property
    def toll(self) -> float:
        return self.my_toll


#########################################################################
def objective_function(x: np.array, alpha: int) -> float:
    size = np.size(x)  # -> 10
    if size <= 1:
        raise Exception("x must be atleast 2-dimension vector")
    sum_values = np.arange(1, size + 1)  # [1,2 ... 10]
    alphas_vector = alpha ** ((sum_values - 1) / (size - 1))
    final_vector = np.square(x) * alphas_vector
    result = np.sum(final_vector)
    return result


#########################################################################


def solver(
    func: callable,
    x0: np.array,
    params: optim_params,
    condition_toggle: bool = True,
    stop_toggle: bool = True,
) -> optim_result:
    my_result = optim_result(params.learn_rate)

    gradient = grad(func)
    # gradient = nd.Gradient(func,0.1)

    learn_rate = params.learn_rate
    iter_info = params.max_iter

    for iteration in range(params.max_iter):
        previous_func_val = func(x0)

        my_result.add_iter(iteration)
        my_result.add_func_value(previous_func_val)

        new_x = x0 - (learn_rate * gradient(x0))
        x0 = new_x  # x update

        if condition_toggle:
            if (
                abs(previous_func_val - func(new_x)) < params.toll
                or np.linalg.norm(gradient(new_x)) < params.toll
            ):
                iter_info = iteration
                stop_toggle = False
                break

    my_result.set_iter_stop(iter_info)
    my_result.set_stop_info_toggle(stop_toggle)

    return my_result

def comparision():

    my_max_iter = 5000  # iteration limit
    my_toll = 0.0001  # stopper ( Elipse value )
    array = np.random.uniform(-100.0, 100.0, (1,10))

    parameters = optim_params(0.1, my_max_iter, my_toll)
    output = solver(f2, array, parameters, False)
    print(output)


def main():
    objective_function_alfa = partial(objective_function, alpha=1)

    my_max_iter = 5000  # iteration limit
    my_toll = 0.0001  # stopper ( Elipse value )
    array = np.random.uniform(100.0, 100.0, size=10)
    print(array)

    parameters = optim_params(0.1, my_max_iter, my_toll)
    output = solver(objective_function_alfa, array, parameters, False)
    print(output)
    pass


if __name__ == "__main__":
    # main()
    comparision()
