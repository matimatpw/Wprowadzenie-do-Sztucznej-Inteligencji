from solver import objective_function, solver, optim_params, optim_result
from functools import partial
import numpy as np
from matplotlib import pyplot as plt


objective_function_alpha_1 = partial(objective_function, alpha=1)
objective_function_alpha_10 = partial(objective_function, alpha=10)
objective_function_alpha_100 = partial(objective_function, alpha=100)


def graph(result: optim_result, alhpa):
    plt.plot(result.get_iterations, result.get_func_values, label=f"Beta = {result.get_beta}")

    plt.xlabel("Iterations")
    plt.ylabel("Function values")
    plt.title(f"Beta influence on function for alpha ={alhpa}")
    plt.legend()
    plt.show()


graph_alfa_1 = partial(graph, alpha=1)
graph_alfa_10 = partial(graph, alpha=10)
graph_alfa_100 = partial(graph, alpha=100)

def main():
    x = np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.])
    x= x *100.
    optim_param_beta_01 = optim_params(0.1,1000,0.0001)
    optim_param_beta_001 = optim_params(0.01,1000,0.0001)
    optim_param_beta_0001 = optim_params(0.001,1000,0.0001)

    result_1 = solver(objective_function_alpha_1, x, optim_param_beta_01)
    result_2 = solver(objective_function_alpha_1, x, optim_param_beta_001)
    result_3 = solver(objective_function_alpha_1, x, optim_param_beta_0001)
#
    graph(result_1,alhpa=100)
    graph(result_2,alhpa=100)
    graph(result_3,alhpa=100)


main()








# categories = ["Category A", "Category B", "Category C", "Category D"]
# values = [10, 24, 16, 30]

# plt.bar(categories, values)

# plt.xlabel('Categories')
# plt.ylabel('Values')
# plt.title('Bar Chart Example')

# plt.show()