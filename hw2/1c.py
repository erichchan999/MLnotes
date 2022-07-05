import numpy as np

start = np.array([-1.2,1])
v = start

def nab(x, y):
    return np.array([-400*(y-(x**2))*x - 2*(1-x), 200*(y-x**2)])

def hes(x, y):
    return np.array([[1200*(x**2)-400*y+2, -400*x], [-400*x, 200]])

new_nab = nab(v[0], v[1])

print(f'k=0, {v}')

iteration = 1
while not np.linalg.norm(new_nab, ord=2) <= 10**(-6):
    new_nab = nab(v[0], v[1])
    new_hes = hes(v[0], v[1])
    v = v - np.linalg.inv(new_hes) @ new_nab
    print(f'k={iteration}, {v}')
    iteration += 1