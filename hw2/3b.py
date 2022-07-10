import numpy as np
import matplotlib.pyplot as plt

SD = 1

def MLE_bias_estimator(n):
    return (2*SD**4)*(n-1)/n**2

def MLE_var_estimator(n):
    return SD**2/n

def new_bias_estimator(n):
    return 0

def new_var_estimator(n):
    return (2*SD**4)/(n-1)

MLE_bias_estimator = np.vectorize(MLE_bias_estimator)
MLE_var_estimator = np.vectorize(MLE_var_estimator)
new_bias_estimator = np.vectorize(new_bias_estimator)
new_var_estimator = np.vectorize(new_var_estimator)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,10))

fig.suptitle('Comparing bias and variance of MLE and new estimator')

def plot(xrange, func1, func2, ylabel, title, x):
    axes[x].plot(xrange, func1(xrange), color='red', label='MLE')
    axes[x].plot(xrange, func2(xrange), color='blue', label='new')
    axes[x].set_title(title)
    axes[x].set_xlabel('n')
    axes[x].set_ylabel(ylabel)
    axes[x].legend(loc='upper right')

xrange = np.arange(2, 201, dtype=np.longdouble)

plot(xrange, MLE_bias_estimator, new_bias_estimator, 'bias', 'bias of MLE and new estimators', 0)
plot(xrange, MLE_var_estimator, new_var_estimator, 'variance', 'variance of MLE and new estimators', 1)

plt.savefig("3b.png", dpi=300)  
plt.show()
