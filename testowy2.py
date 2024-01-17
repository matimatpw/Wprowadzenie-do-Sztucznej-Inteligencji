import matplotlib.pyplot as plt
import numpy as np

# Example data
x = [1,2,3,4,5,6,7,8,9,10]
y = [10,20,30,40,50,60,70,80,90,100]
sizes = np.random.randint(10, 100, 50)  # Random sizes for each point

# Scatter plot with custom sizes
# plt.scatter(x, y, s=2, alpha=0.8)

# Connect the dots with lines
plt.plot(x, y, linestyle='-', color='blue', alpha=0.5)

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot with Connected Dots')
plt.show()
