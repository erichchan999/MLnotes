from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
import joblib

print('Loading train data...')
train_data = np.loadtxt('train_data.csv', delimiter=',')
train_labels = np.loadtxt('train_labels.csv', delimiter=',')
print('Finished loading train data')

tree = DecisionTreeClassifier(random_state=0, class_weight='balanced') 
# What does class_weight do? Solution to imbalanced dataset (number of pneumonia positive cases >>> negative cases)
# https://datascience.stackexchange.com/questions/56250/how-does-class-weight-work-in-decision-tree
# https://stackoverflow.com/questions/37522191/how-to-balance-classification-using-decisiontreeclassifier

# Why is there randomness in decision trees? Multiple trees are trained in an ensemble learner, where features and samples are randomly sampled with replacement
# https://stackoverflow.com/questions/39158003/confused-about-random-state-in-decision-tree-of-scikit-learn

print('Fitting decision tree...')
tree = tree.fit(train_data, train_labels)
print('Finished fitting decsion tree')

print('Saving decision tree to cwd...')
joblib.dump(tree, 'dtree.joblib')
print('Decision tree saved to cwd')



