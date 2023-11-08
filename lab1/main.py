from solver import objective_function, solver, optim_params, optim_result
from functools import partial
import numpy as np
from matplotlib import pyplot as plt
import time
import json
import warnings


class objective_functions_factory:
    def __init__(self) -> None:
        self._objective_function_alpha_1 = self.create_objective_function(1)
        self._objective_function_alpha_10 = self.create_objective_function(10)
        self._objective_function_alpha_100 = self.create_objective_function(100)

    def create_objective_function(self, my_alhpa: int) -> partial:
        return partial(objective_function, alpha=my_alhpa)

    @property
    def alpha_1(self) -> partial:
        return self._objective_function_alpha_1

    @property
    def alpha_10(self) -> partial:
        return self._objective_function_alpha_10

    @property
    def alpha_100(self) -> partial:
        return self._objective_function_alpha_100


# objective_function_alpha_1 = partial(objective_function, alpha=1)
# objective_function_alpha_10 = partial(objective_function, alpha=10)
# objective_function_alpha_100 = partial(objective_function, alpha=100)


class result:
    def __init__(
        self,
        beta_01: optim_params,
        beta_001: optim_params,
        beta_0001: optim_params,
        x: np.array,
        obj_fun: objective_functions_factory,
    ) -> None:
        self.beta_01 = beta_01
        self.beta_001 = beta_001
        self.beta_0001 = beta_0001
        self.x = x
        self.obj_func = obj_fun

        self.results_alfa1 = []
        self.results_alfa10 = []
        self.results_alfa100 = []

    def create_results_beta01(
        self,
        obj_func_alpha: partial,
        result_list: [optim_result],
        my_time: {},
        alpha_str: str,
    ) -> None:
        time_start = time.time()
        my_result = solver(obj_func_alpha, self.x, self.beta_01)

        time_stop = time.time()
        result_list.append(my_result)
        my_time[alpha_str]["Beta=0.1"] = time_stop - time_start

    def create_results_beta001(
        self,
        obj_func_alpha: partial,
        result_list: [optim_result],
        my_time: {},
        alpha_str: str,
    ) -> None:
        time_start = time.time()
        my_result = solver(obj_func_alpha, self.x, self.beta_001)

        time_stop = time.time()
        result_list.append(my_result)
        my_time[alpha_str]["Beta=0.01"] = time_stop - time_start

    def create_results_beta0001(
        self,
        obj_func_alpha: partial,
        result_list: [optim_result],
        my_time: {},
        alpha_str: str,
    ) -> None:
        time_start = time.time()
        my_result = solver(obj_func_alpha, self.x, self.beta_0001)

        time_stop = time.time()
        result_list.append(my_result)
        my_time[alpha_str]["Beta=0.001"] = time_stop - time_start


def graph(my_result_list: [optim_result], filename: str, alpha: int) -> None:
    for result in my_result_list:
        print(result)
        plt.scatter(
            result.get_iterations(),
            result.get_func_values(),
            label=f"Beta = {result.learn_info}",
        )

    plt.xlabel("Iterations")
    plt.ylabel("Function values")
    plt.title(f"Beta influence on function for alpha ={alpha}")
    plt.legend()
    plt.savefig(filename, format="pdf")


def graph_each_beta(
    my_result_list: [optim_result], base_filename: str, alpha: int
) -> None:
    for i, result in enumerate(my_result_list):
        print(result)
        plt.scatter(
            result.get_iterations(),
            result.get_func_values(),
            label=f"Beta = {result.learn_info}",
        )

        plt.xlabel("Iterations")
        plt.ylabel("Function values")
        plt.title(f"Beta influence on function for alpha ={alpha}")
        plt.legend()
        filename = f"{base_filename}_{i}.pdf"
        plt.savefig(filename, format="pdf")
        plt.close()


graph_alfa_1 = partial(graph, alpha=1)
graph_alfa_10 = partial(graph, alpha=10)
graph_alfa_100 = partial(graph, alpha=100)
graph_alfa100_each_beta = partial(graph_each_beta, alpha=100)


def main() -> None:
    x = np.random.uniform(-100.0, 100.0, size=10)

    obj_function = objective_functions_factory()

    optim_param_beta_01 = optim_params(0.1, 1000, 0.0001)
    optim_param_beta_001 = optim_params(0.01, 1000, 0.0001)
    optim_param_beta_0001 = optim_params(0.001, 1000, 0.0001)

    my_result = result(
        optim_param_beta_01,
        optim_param_beta_001,
        optim_param_beta_0001,
        x,
        obj_function,
    )
    my_time = {
        "alpha1": {"Beta=0.1": None},
        "alpha10": {"Beta=0.01": None},
        "alpha100": {"Beta=0.001": None},
    }

    #    ## results for alpha = 1
    my_result.create_results_beta01(
        obj_function.alpha_1, my_result.results_alfa1, my_time, "alpha1"
    )
    my_result.create_results_beta001(
        obj_function.alpha_1, my_result.results_alfa1, my_time, "alpha1"
    )
    my_result.create_results_beta0001(
        obj_function.alpha_1, my_result.results_alfa1, my_time, "alpha1"
    )
    graph_alfa_1(my_result.results_alfa1, "alfa_1_betas.pdf")

    #   ## results for alpha = 10
    # my_result.create_results_beta01(
    #     obj_function.alpha_10, my_result.results_alfa10, my_time, "alpha10"
    # )
    # my_result.create_results_beta001(
    #     obj_function.alpha_10, my_result.results_alfa10, my_time, "alpha10"
    # )
    # my_result.create_results_beta0001(
    #     obj_function.alpha_10, my_result.results_alfa10, my_time, "alpha10"
    # )
    # graph_alfa_10(my_result.results_alfa10, "alfa_10_betas.pdf")

    # results for alpha = 100
    # my_result.create_results_beta01(
    #     obj_function.alpha_100, my_result.results_alfa100, my_time, "alpha100"
    # )
    # my_result.create_results_beta001(
    #     obj_function.alpha_100, my_result.results_alfa100, my_time, "alpha100"
    # )
    # my_result.create_results_beta0001(
    #     obj_function.alpha_100, my_result.results_alfa100, my_time, "alpha100"
    # )
    # graph_alfa_100(my_result.results_alfa100, "alfa_100_betas.pdf") # --> weird plot
    # graph_alfa100_each_beta(my_result.results_alfa100, "alfa_100_betas.pdf")

    print(my_time)

    with open("solver_times.json", "w") as file:
        json.dump(my_time, file, indent=4)


if __name__ == "__main__":
    main()
