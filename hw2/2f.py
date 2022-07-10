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


def hes_loss(gamma, X, y, lam):
    n = X.shape[0]

    summ = np.zeros((p+1, p+1))
    for i in range(0,n):
        # appending the extra 1 as first element of each x datapoint
        xi = np.insert(X[i], 0, 1)
        summ += sigmoid(np.dot(gamma, xi)) * (1 - sigmoid(np.dot(gamma, xi))) * np.outer(xi, np.transpose(xi))

    return (lam / n) * summ


p = data.X_train.shape[1]
gd_gamma = np.zeros(p + 1)
newt_gamma = np.zeros(p + 1)
lam = 0.5
alpha = 1
a = 0.5
b = 0.8

epochs_lim = 3
epochs = np.arange(1, epochs_lim+1)
step_sizes = np.full(epochs_lim, -1.0)
gd_losses = np.full(epochs_lim, -1.0)
newt_losses = np.full(epochs_lim, -1.0)


# Log regression with gradient descent
for ep in range(1, epochs_lim+1):
    cur_nab_loss = nab_loss(gd_gamma, data.X_train, data.Y_train, lam)
    cur_loss = loss(gd_gamma, data.X_train, data.Y_train, lam)
    
    if loss(gd_gamma - alpha * cur_nab_loss, data.X_train, data.Y_train, lam) > cur_loss - a * alpha * np.linalg.norm(cur_nab_loss, ord=2)**2:
        alpha = alpha * b

    step_sizes[ep-1] = alpha

    # Update equation
    gd_gamma = gd_gamma - alpha * cur_nab_loss

    gd_losses[ep-1] = loss(gd_gamma, data.X_train, data.Y_train, lam)


# Log regression with newton's method
for ep in range(1, epochs_lim+1):
    cur_nab_loss = nab_loss(newt_gamma, data.X_train, data.Y_train, lam)
    cur_loss = loss(newt_gamma, data.X_train, data.Y_train, lam)
    
    if loss(newt_gamma - alpha * cur_nab_loss, data.X_train, data.Y_train, lam) > cur_loss - a * alpha * np.linalg.norm(cur_nab_loss, ord=2)**2:
        alpha = alpha * b

    step_sizes[ep-1] = alpha

    # Update equation
    newt_gamma = newt_gamma - alpha * (np.linalg.inv(hes_loss(newt_gamma, data.X_train, data.Y_train, lam)) @ cur_nab_loss)

    newt_losses[ep-1] = loss(newt_gamma, data.X_train, data.Y_train, lam)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,10))

axes[0].scatter(epochs, step_sizes)
axes[0].set_xlabel('epoch')
axes[0].set_ylabel('step size')
axes[0].set_title('Change of step size over each epoch')

axes[1].scatter(epochs, gd_losses, color='red', label='gradient descent')
axes[1].scatter(epochs, newt_losses, color='blue', label='newton\'s method')
axes[1].set_xlabel('epoch')
axes[1].set_ylabel('loss')
axes[1].set_title('Comparing GD and newtown\'s method on log-loss')
axes[1].legend(loc='upper right')

print('final train loss:', loss(newt_gamma, data.X_train, data.Y_train, lam))
print('test loss:', loss(newt_gamma, data.X_test, data.Y_test, lam))

plt.savefig("2f.png", dpi=300)  
plt.show()
