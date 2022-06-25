import pandas as pd
from sklearn.preprocessing import StandardScaler

features = pd.read_csv('features.csv')

scaler = StandardScaler()
features = scaler.fit_transform(features)

print(scaler.mean_)
print(scaler.var_)

target = pd.read_csv('target.csv')

target = target - target.mean()

X_train = features[:features.shape[0]//2]
X_test = features[features.shape[0]//2:]
Y_train = target[:target.shape[0]//2]
Y_test = target[target.shape[0]//2:]

print(X_train[0:3])
print(X_train[-3:])
print(X_test[0:3])
print(X_test[-3:])
print(Y_train[0:3])
print(Y_train[-3:])
print(Y_test[0:3])
print(Y_test[-3:])
