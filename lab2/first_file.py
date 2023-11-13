

# import numpy as np
# from cec2017 import functions
# x_range = np.linspace(-100, 100, 10)
# y_range = np.linspace(-100, 100, 10)

# X, Y = np.meshgrid(x_range, y_range)


# Z2 = np.empty(X.shape)
# for i in range(X.shape[0]):
#     for j in range(X.shape[1]):
#         x = np.array([[X[i, j], Y[i, j]]])
#         # Z2[i, j] = functions.f1(x)


# # print(Z2.shape)
# xd = np.array([[-100., -100.,10]])






# [[-77.67696192, -77.67696192, -77.67696192, -77.67696192,
#         -77.67696192, -77.67696192, -77.67696192, -77.67696192,
#         -77.67696192, -77.67696192]]


# x = np.array([[0, 0, 0, 0, 0,0,0,0,0,0]])
# print(x.shape)
# x =np.random.uniform(-100.0, 100.0, size=10)
# print(x)

# # if iteration % control_rate == 0: # teraz sprawdzamy czy w przeciagu <a> iteracji
# #             if float(success) / float(control_rate) > 0.3:
# #                 step_size *= 1.3
# #             if float(success) / float(control_rate) < 0.3:
# #                 step_size *= 0.3
# #             success = 0
import random
import numpy as np
x0 = np.random.uniform(100.,100.,(1,10))
# x0 = x0.reshape(1,-1)
print(x0)
y = np.array([val + 1.0 * random.normalvariate(0., 1.) for val in x0])
# print(x0.shape[1])

# x = np.array([[(x0 + 1.0 * random.normalvariate(0.,1.)) for _ in range(x0.shape[1])]])



# x = np.array([[(x0 * random.normalvariate(0.,1.)) for _ in range(10)]])
# print(x)
# y = [random.normalvariate(0.,1.) for _ in x]
# y = [val + 0.22 * random.normalvariate(0.0, 1.) for val in x]
# print(y)



# print()

# print(random.normalvariate(0.,1.))
# print(np.random.normal(0.,1.,(1,10)))

# print()



import numpy as np

# # Create a random array of shape (1, 10)
# array1 = np.random.uniform(0.0, 1.0, (1, 10))

# # Create another array of shape (1, 10) or any compatible shape
# array2 = np.random.uniform(1.0, 2.0, (1, 10))
# # print(array1.shape)
# # Multiply each element of array1 by the corresponding element of array2
# result_array = array1 + array2

# # Print the result
# print(result_array)

# print("XDDDD")

# my_x = np.random.uniform(100.,100., (1, 10))

# my_y = np.random.normal(0.,1.,(1,10))
# my_y *= 1.0
# my_y = my_x + my_y
# last_100_elements = my_y[-50:]
# if np.all(last_100_elements) == last_100_elements[0]:

# print(xxx)
# import random
# import numpy as np

# from cec2017_test import F1, f1, f9

# #odpalic pare razy algorytm na tych samych danych poczatkowych

# class optim_params:
#     def __init__(self) -> None:
#         self.my_control_rate = 5
#         self.my_step_size = 1.0
#         self.my_max_iterations = 1000

#         self.my_success = 0
#         #hard coded -> make parametrized
#         self.my_mean = 0.0
#         self.my_deviation = 1.0

# #step_size: float, control_rate: int, max_iter
#     @property
#     def control_rate(self):
#         return self.my_control_rate

#     @property
#     def success(self):
#         return self.my_success

#     @property
#     def step_size(self):
#         return self.my_step_size

#     @property
#     def max_iter(self):
#         return self.my_max_iterations

#     @property
#     def deviation(self):
#         return self.my_deviation

#     @property
#     def mean(self):
#         return self.my_mean

#     def set_stepsize(self, new_step)-> None:
#         self.my_step_size = new_step

#     def set_success(self, new_success)-> None:
#         self.my_success = new_success




# def stepsize_adaptation(iteration: int, o_p: optim_params):
#     if iteration % o_p.control_rate == 0: # teraz sprawdzamy czy w przeciagu <a> iteracji
#         if float(o_p.success) / float(o_p.control_rate) > 0.2:
#             o_p.set_stepsize(o_p.step_size * 1.22)
#         if float(o_p.success) / float(o_p.control_rate) < 0.2:
#             o_p.set_stepsize(o_p.step_size * 0.22)
#         o_p.set_success(0)
#     return o_p


# def evolution_1p1(f:callable, x0: [float], o_p:optim_params, stepsize_adaptation: callable):
#     x = x0.copy()

#     for iteration in range(o_p.max_iter):
# # MUTATION (metoda Gaussa)
#         y = np.array([val + o_p.step_size * random.normalvariate(o_p.mean, o_p.deviation) for val in x])
# #TODO correct y_array so that it randoms each value in x_array ( now it applies the same value for ale the elements in x_arr!!!)
# # EVALUATE OFFSPRING
#         if f(y) <= f(x): # poprawa potomka
#             o_p.set_success(o_p.success +1)
#             x = y
# #UPDATING mutation_rate based on actual status


#         o_p = stepsize_adaptation(iteration, o_p)

#         fx_value = f(x)
#         # if(iteration == 999):
#         print(f"iteration: {iteration},\tvalue: {fx_value}")
#     return [x, f(x)]


# #--GENEROIWAnie populacji--#
# x_range = np.linspace(-100, 100, 10)
# y_range = np.linspace(-100, 100, 10)
# # print(x_range)
# X, Y = np.meshgrid(x_range, y_range)


# # x = np.column_stack((X.ravel(), Y.ravel()))
# # print(x.shape)



# x = np.array([[np.random.uniform(100.0, 100.0) for _ in range(10)]])
# # x = np.array([[X[1,1], Y[1,1],X[1,1], Y[1,1],X[1,1], Y[1,1],X[1,1], Y[1,1],X[1,1], Y[1,1]]])        ### DLA WYMIAROWOSCI 10 CZYLI [[X[1,1], Y[1,1], Z ... Z10[1,1]  ]]
# x = np.array([[X[1,1], Y[1,1],X[1,1], Y[1,1],X[1,1], Y[1,1],X[1,1], Y[1,1],X[1,1], Y[1,1]]])
# print(x)
# # print(x.shape)

# # x = np.empty((0,2))
# # for i in range(X.shape[0]):
# #     for j in range(X.shape[1]):
# #         x = np.vstack((x, [X[i, j], Y[i, j]]))                                                               ### DLA WYMIAROWOSCI 10 CZYLI [[X[1,1], Y[1,1], Z ... Z10[1,1]  ]]


# #TODO czy wykonac program i przy nastepnym uruchomieniu algorytm na korzystac z poprzedniej populacji?
# #TODO czy wnioskowac jak algorytm sie szachowuje przy kazdym kolejnym uruchomieniu
# #TODO jak z ta 10 wymiarowoscia funckji?


# my_optim_params = optim_params()


# # print(f1(x))
# # print(f9(x))
# list_xd = evolution_1p1(f1,x,my_optim_params,stepsize_adaptation)
# print(list_xd)
# # print(list_xd[0])
# print("____________________________\n")
# # print(evolution_1p1(f9,list_xd[0],1.0,10,100))
# #start-point 100... / random

def calculate_average(*arrays) -> list:
    return [sum(values) / len(arrays) for values in zip(*arrays)]

a = [1, 2, 3, 4]
b = [5, 10, 20, 30]
c = [3, 1, 1, 1]

# Add more arrays as needed
# d = [x, y, z, ...]

# Calculate the average for each position
final = calculate_average(a, b, c)

print(final)