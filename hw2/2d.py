import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


df=pd.read_csv('songs.csv', sep=',', header=0)


# I
df.drop(columns=['Artist Name', 'Track Name', 'key', 'mode', 'time_signature', 'instrumentalness'], inplace=True)


# II
df.drop(df[(df.Class != 5) & (df.Class != 9)].index, inplace=True)
df['Class'].replace([5, 9], [1, 0], inplace=True)


# III
df.dropna(axis=0, inplace=True)


# IV
X_train, X_test, Y_train, Y_test = train_test_split(df.drop(columns='Class'), df['Class'], test_size=0.3, random_state=23)


# V
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# VI
Y_train = Y_train.to_numpy()
Y_test = Y_test.to_numpy()

if __name__ == '__main__':
    print('first row X_train:',X_train[0][0:3])
    print('last row X_train:',X_train[-1][0:3])
    print('first row X_test:',X_test[0][0:3])
    print('last row X_test:',X_test[-1][0:3])
    print('first row Y_train:',Y_train[0])
    print('last row Y_train:',Y_train[-1])
    print('first row Y_test:',Y_test[0])
    print('last row Y_test:',Y_test[-1])