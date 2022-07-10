import numpy as np
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
data = __import__('2d')

def sigmoid(x):
    # logistic sigmoid
    return np.exp(-np.logaddexp(0, -x))

def loss(gamma, X, y, lam):
    # gamma has first coordinate = beta0 = intercept, and second coordinate = beta
    norm_beta_sq = np.linalg.norm(gamma[1:], ord=2)**2
    z = np.dot(X, gamma[1:]) + gamma[0]
    sig_z = sigmoid(z)
    return lam * log_loss(y, sig_z, normalize=True) + 0.5 * norm_beta_sq

def nab_loss(gamma, X, y, lam):
    n = X.shape[0]

    summ = np.zeros(p+1)
    for i in range(0,n):
        summ += (y[i] - sigmoid(np.dot(gamma, np.insert(X[i], 0, 1)))) * np.insert(X[i], 0, 1)
    
    return np.insert(gamma[1:], 0, 0) - (lam / n) * summ

p = data.X_train.shape[1]
gamma = np.zeros(p + 1)
lam = 0.5
alpha = 1
a = 0.5
b = 0.8

epochs_lim = 60
epochs = np.arange(1, epochs_lim+1)
step_sizes = np.full(epochs_lim, -1.0)
losses = np.full(epochs_lim, -1.0)

for ep in range(1, epochs_lim+1):
    cur_nab_loss = nab_loss(gamma, data.X_train, data.Y_train, lam)
    cur_loss = loss(gamma, data.X_train, data.Y_train, lam)
    
    if loss(gamma - alpha * cur_nab_loss, data.X_train, data.Y_train, lam) > cur_loss - a * alpha * np.linalg.norm(cur_nab_loss, ord=2)**2:
        alpha = alpha * b

    step_sizes[ep-1] = alpha

    # Update equation
    gamma = gamma - alpha * cur_nab_loss

    losses[ep-1] = loss(gamma, data.X_train, data.Y_train, lam)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,10))

def plot(x, y, xlabel, ylabel, title, loc):
    axes[loc].scatter(x, y)
    axes[loc].set_title(title)
    axes[loc].set_xlabel(xlabel)
    axes[loc].set_ylabel(ylabel)

plot(epochs, step_sizes, 'epoch', 'step size', 'Change of step size over each epoch', 0)
plot(epochs, losses, 'epoch', 'loss', 'Change of loss over each epoch', 1)

print('final train loss:', loss(gamma, data.X_train, data.Y_train, lam))
print('test loss:', loss(gamma, data.X_test, data.Y_test, lam))

plt.savefig("2e.png", dpi=300)  
plt.show()
