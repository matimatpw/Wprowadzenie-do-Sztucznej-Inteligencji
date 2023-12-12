import numpy as np

# Assuming y[index] has shape (411, 1)
y_index = np.random.rand(411, 1)

# Assuming (self.alpha * y).dot(self.kernel(X, X[index])) has shape (649, 411)
alpha_y_kernel = np.random.rand(649, 411)

# Transpose y[index] to make its shape (1, 411)
y_index_transposed = y_index.T

# Subtract the arrays
result = y_index_transposed - alpha_y_kernel

print(y_index)
print(alpha_y_kernel.size)
print(y_index_transposed.size)
print(266739/649)