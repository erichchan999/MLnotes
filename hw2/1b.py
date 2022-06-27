import numpy as np
import matplotlib.pyplot as plt

def func(x,y):
    return 100*(y-x^2)^2 + (1-x)^2

# create two one-dimensional grids using linspace
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)

# combine the two one-dimensional grids into one two-dimensional grid
X, Y = np.meshgrid(x,y)

# evaluate the function at each element of the two-dimensional grid
Z = func(X, Y)

# create plot
fig = plt.figure(figsize=(7,7))
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
plt.show()