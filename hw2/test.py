import numpy as np

x=10
y=-10

def hes(x, y):
    return np.array([[1200*(x**2)-400*y+2, -400*x], [-400*x, 200]])

print(hes(x, y))
