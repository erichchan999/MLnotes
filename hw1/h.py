import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

features = pd.read_csv('features.csv')

scaler = StandardScaler()
features = scaler.fit_transform(features)

target = pd.read_csv('target.csv')

target = target - target.mean()

X_train = pd.DataFrame(features[:features.shape[0]//2]).to_numpy()
X_test = pd.DataFrame(features[features.shape[0]//2:]).to_numpy()
Y_train = pd.DataFrame(target[:target.shape[0]//2]).to_numpy()
Y_test = pd.DataFrame(target[target.shape[0]//2:]).to_numpy()
Y_train = np.reshape(Y_train, -1)
Y_test = np.reshape(Y_test, -1)

X_train_transpose = X_train.transpose()

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10,10))
fig.tight_layout() 

phi = 0.5
n = X_train.shape[0]
p = X_train.shape[1]

betahat = np.linalg.inv(X_train_transpose @ X_train + phi * np.eye(p)) @ X_train_transpose @ Y_train

def loss_func(beta):
    return (1/n) * (np.linalg.norm(Y_train - X_train @ beta, ord=2) ** 2) + (np.linalg.norm(beta, ord=2) ** 2)

def gradient_i(beta, i):
    return 2 * ((np.dot(X_train[i], beta) * X_train[i]) - (Y_train[i] * X_train[i]) + (1/n) * phi * beta)

def MSE_train(beta):
    return (1/n) * (np.linalg.norm(Y_train - X_train @ beta, ord=2) ** 2)

def MSE_test(beta):
    return (1/n) * (np.linalg.norm(Y_test - X_test @ beta, ord=2) ** 2)

r_plot = 0
c_plot = 0

epochs = 5*n

results = []

for alpha in [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.006, 0.02]:
    indexes = np.arange(epochs)
    deltas = np.full(epochs, -1.0)

    start = np.ones(p)
    beta_k = start

    for i in range(0, epochs):
        gradient_beta_i = gradient_i(beta_k, i % n)
        beta_k = beta_k - alpha * gradient_beta_i
        
        delta_k = loss_func(beta_k) - loss_func(betahat)
        deltas[i] = delta_k

    if alpha == 0.006:
        print(MSE_train(beta_k))
        print(MSE_test(beta_k))

    axes[r_plot, c_plot].scatter(indexes, deltas)
    axes[r_plot, c_plot].set_title(f'alpha = {alpha}')
    axes[r_plot, c_plot].set_xlabel('epoch * 200 (size of sample)')
    axes[r_plot, c_plot].set_ylabel('delta')

    if c_plot >= 2:
        r_plot += 1
        c_plot = 0
    else:
        c_plot += 1

plt.show()
