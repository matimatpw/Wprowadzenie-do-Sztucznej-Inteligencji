
import matplotlib.pyplot as plt
import cec2017
import numpy as np
from cec2017_test import wrapper_for_1d_function

from cec2017.functions import f1

def F1(x):
    return f1(wrapper_for_1d_function(x))

MAX_X = 1
PLOT_STEP = 0.1

x_arr = np.arange(-MAX_X, MAX_X, PLOT_STEP)
y_arr = np.arange(-MAX_X, MAX_X, PLOT_STEP)
print(x_arr)
X, Y = np.meshgrid(x_arr, y_arr)
Z = np.empty(X.shape)

q=F1


for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        print("XD\n")
        Z[i, j] = q(np.array([X[i, j], Y[i, j]]))
        
plt.contour(X, Y, Z, 20)
plt.arrow(0, 0, 50, 50, head_width=3, head_length=6, fc='k', ec='k')
plt.savefig("jest.pdf")
  