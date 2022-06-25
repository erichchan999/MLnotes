import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

features = pd.read_csv('features.csv')

scaler = StandardScaler()
features = scaler.fit_transform(features)

target = pd.read_csv('target.csv')

target = target - target.mean()

X_train = features[:features.shape[0]//2]
X_test = features[features.shape[0]//2:]
Y_train = target[:target.shape[0]//2]
Y_test = target[target.shape[0]//2:]

phi = 0.5

ridge_regression_weights = np.linalg.inv(X_train.transpose() @ X_train + phi * np.eye(X_train.shape[1])) @ X_train.transpose() @ Y_train
print(ridge_regression_weights)