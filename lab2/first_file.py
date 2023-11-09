

import numpy as np
from cec2017 import functions
x_range = np.linspace(-100, 100, 10)
y_range = np.linspace(-100, 100, 10)

X, Y = np.meshgrid(x_range, y_range)


Z2 = np.empty(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x = np.array([[X[i, j], Y[i, j]]])
        # Z2[i, j] = functions.f1(x)


# print(Z2.shape)
xd = np.array([[-100., -100.,10]])
print(xd.shape)





[[-77.67696192, -77.67696192, -77.67696192, -77.67696192,
        -77.67696192, -77.67696192, -77.67696192, -77.67696192,
        -77.67696192, -77.67696192]]


x = np.array([[0, 0, 0, 0, 0,0,0,0,0,0]])
print(x.shape)
x =np.random.uniform(-100.0, 100.0, size=10)
print(x)