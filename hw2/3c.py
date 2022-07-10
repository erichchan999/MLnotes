import numpy as np
import matplotlib.pyplot as plt

SD = 1

def MLE_estimator_MSE(n):
    return (2*SD**4/n) - (SD**4/n**2)

def new_estimator_MSE(n):
    return (2*SD**4) / (n-1)

MLE_estimator_MSE = np.vectorize(MLE_estimator_MSE)
new_estimator_MSE = np.vectorize(new_estimator_MSE)

xrange = np.arange(2, 101, dtype=np.longdouble)

plt.plot(xrange, MLE_estimator_MSE(xrange), color='red', label='MLE estimator')
plt.plot(xrange, new_estimator_MSE(xrange), color='blue', label='new estimator')

plt.title('Comparing MSE of MLE and new estimator')
plt.legend(loc='upper right')
plt.xlabel('n')
plt.ylabel('MSE')
plt.ylim(0, 2.2)

plt.savefig("3c.png", dpi=300)  
plt.show()
