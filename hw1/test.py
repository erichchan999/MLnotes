import numpy as np

X_train = np.array([[1,2,3], [4,5,6]])
Y_train = np.array([[8,9]])
beta = np.ones(3)
print(beta)

# def gradient_i(beta, i):
    # return 2 * (((X_train[i][:].reshape(1,3) @ beta) * X_train[i][:].reshape(3,1)) - (Y_train[i][0] * X_train[i][:]))

print(np.dot(X_train[0], X_train[0]) * beta)
print(np.dot(X_train[0], beta) * X_train[0])
# print(gradient_i(np.array([[5,5,5]]), 0))