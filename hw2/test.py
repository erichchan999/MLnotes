import numpy as np


a=np.array([[1, 2, 3]]).flatten()
b=np.array([[3, 4], [5, 6]])
print(np.outer(a, np.transpose(a)))
# print(5*a)