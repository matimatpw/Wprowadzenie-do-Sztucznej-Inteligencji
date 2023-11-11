import random
import numpy as np

from cec2017_test import F1, f1, f9

#odpalic pare razy algorytm na tych samych danych poczatkowych

class optim_params:
    def __init__(self) -> None:
        self.my_control_rate = 5
        self.my_step_size = 1.0
        self.my_max_iterations = 10000

        self.my_success = 0
        #hard coded -> make parametrized
        self.my_mean = 0.0
        self.my_deviation = 1.0

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

    def set_stepsize(self, new_step)-> None:
        self.my_step_size = new_step

    def set_success(self, new_success)-> None:
        self.my_success = new_success




def stepsize_adaptation(iteration: int, o_p: optim_params):
    if iteration % o_p.control_rate == 0: # teraz sprawdzamy czy w przeciagu <a> iteracji
        if float(o_p.success) / float(o_p.control_rate) > 0.2:
            o_p.set_stepsize(o_p.step_size * 1.22)
        if float(o_p.success) / float(o_p.control_rate) < 0.2:
            o_p.set_stepsize(o_p.step_size * 0.22)
        o_p.set_success(0)
    return o_p


def evolution_1p1(f:callable, x0: [float], o_p:optim_params, stepsize_adaptation: callable):
    x = x0.copy()

    for iteration in range(o_p.max_iter):
# MUTATION (metoda Gaussa)
        # print(x)
        # y = np.array([val + o_p.step_size * random.normalvariate(o_p.mean, o_p.deviation) for val in x])
        my_y = np.random.normal(0.,1.,(1,10))
        my_y *= 1.0
        y = x + my_y
        # print(y)
        # return
#TODO correct y_array so that it randoms each value in x_array ( now it applies the same value for ale the elements in x_arr!!!)

# EVALUATE OFFSPRING
        if f(y) <= f(x): # poprawa potomka
            o_p.set_success(o_p.success +1)
            x = y
#UPDATING mutation_rate based on actual status


        o_p = stepsize_adaptation(iteration, o_p)

        fx_value = f(x)
        # if(iteration == 999):
        print(f"iteration: {iteration},\tvalue: {fx_value}")
    return [x, f(x)]


#--GENEROIWAnie populacji--#
x_range = np.linspace(-100, 100, 10)
y_range = np.linspace(-100, 100, 10)
# print(x_range)
X, Y = np.meshgrid(x_range, y_range)


# x = np.column_stack((X.ravel(), Y.ravel()))
# print(x.shape)



x = np.array([[np.random.uniform(100.0, 100.0) for _ in range(10)]])
# x = np.array([[X[1,1], Y[1,1],X[1,1], Y[1,1],X[1,1], Y[1,1],X[1,1], Y[1,1],X[1,1], Y[1,1]]])        ### DLA WYMIAROWOSCI 10 CZYLI [[X[1,1], Y[1,1], Z ... Z10[1,1]  ]]
# x = np.array([[X[1,1], Y[1,1],X[1,1], Y[1,1],X[1,1], Y[1,1],X[1,1], Y[1,1],X[1,1], Y[1,1]]])
print(x)
# print(x.shape)
print("____________________________\n")
# x = np.empty((0,2))
# for i in range(X.shape[0]):
#     for j in range(X.shape[1]):
#         x = np.vstack((x, [X[i, j], Y[i, j]]))                                                               ### DLA WYMIAROWOSCI 10 CZYLI [[X[1,1], Y[1,1], Z ... Z10[1,1]  ]]


#TODO czy wykonac program i przy nastepnym uruchomieniu algorytm na korzystac z poprzedniej populacji?
#TODO czy wnioskowac jak algorytm sie szachowuje przy kazdym kolejnym uruchomieniu
#TODO jak z ta 10 wymiarowoscia funckji?


my_optim_params = optim_params()


print(evolution_1p1(f9,x,my_optim_params,stepsize_adaptation))

# print(f1(x))
# print(f9(x))
# list_xd = evolution_1p1(f9,x,1.0,10,100)
# print(list_xd)
# print(list_xd[0])
print("____________________________\n")
# print(evolution_1p1(f9,list_xd[0],1.0,10,100))
#start-point 100... / random




