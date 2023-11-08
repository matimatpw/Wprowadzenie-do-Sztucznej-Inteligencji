import random
import numpy as np

from cec2017_test import F1, f1


def evolution_1p1(f, x0: [float], mutation_rate: float, control_rate: int, max_iter):
    x = x0.copy()
    
    mean = 0.0              # srednia arytmetyczna -> wokol takiej wartosci bedziemy losowac
    deviation = 1.0         # odchylenie standardowe
    success = 0
    for iteration in range(max_iter):
# MUTATION (metoda Gaussa)
        y = np.array([val + mutation_rate * random.normalvariate(mean, deviation) for val in x])

# EVALUATE OFFSPRING
        if f(y) <= f(x): # poprawa potomka
            success += 1
            x = y
#UPDATING mutation_rate based on actual status
        if iteration % control_rate == 0: # teraz sprawdzamy czy w przeciagu <a> iteracji
            if float(success) / float(control_rate) > 0.3:
                mutation_rate *= 1.3
            if float(success) / float(control_rate) < 0.3:
                mutation_rate *= 0.3
            success = 0
        fx_value = f(x)
        print(f"iteration: {iteration},\tvalue: {fx_value}")
    return [x, f(x)]


#--GENEROIWAnie populacji--#
x_range = np.linspace(-100, 100, 10)
y_range = np.linspace(-100, 100, 10)
# print(x_range)
X, Y = np.meshgrid(x_range, y_range)


# x = np.column_stack((X.ravel(), Y.ravel()))
# print(x.shape)



# x = np.array([[np.random.uniform(-100.0, 100.0) for _ in range(10)]])
# x = np.array([[X[1,1], Y[1,1],X[1,1], Y[1,1],X[1,1], Y[1,1],X[1,1], Y[1,1],X[1,1], Y[1,1]]])        ### DLA WYMIAROWOSCI 10 CZYLI [[X[1,1], Y[1,1], Z ... Z10[1,1]  ]]
x = np.array([[X[1,1], Y[1,1]]]) 
print(x)

# x = np.empty((0,2))
# for i in range(X.shape[0]):
#     for j in range(X.shape[1]):
#         x = np.vstack((x, [X[i, j], Y[i, j]]))                                                               ### DLA WYMIAROWOSCI 10 CZYLI [[X[1,1], Y[1,1], Z ... Z10[1,1]  ]]


#TODO czy wykonac program i przy nastepnym uruchomieniu algorytm na korzystac z poprzedniej populacji?
#TODO czy wnioskowac jak algorytm sie szachowuje przy kazdym kolejnym uruchomieniu
#TODO jak z ta 10 wymiarowoscia funckji?



# print(f1(x))
print(evolution_1p1(f1,x,1.0,10,1000))



