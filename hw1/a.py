import numpy as np

answers = []

A = np.array([[1,2,1,-1],[-1,1,0,2],[0,-1,-2,1]])
A_transpose = A.transpose()
b = np.array([3,2,-2])
gamma = 0.2
alpha = 0.1

def gradient(x):
    return ((A_transpose @ A @ x) - (A_transpose @ b) + (gamma * x))

start = np.array([1,1,1,1])
x = start
answers.append(np.round_(x.copy(), decimals=4))
gradient_at_x = gradient(x)

while np.linalg.norm(gradient_at_x) >= 0.001:
    x = x - alpha * gradient_at_x
    answers.append(np.round_(x.copy(), decimals=4))
    gradient_at_x = gradient(x)

print(answers[:5])
print(answers[-5:])