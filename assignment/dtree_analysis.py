from sklearn import metrics
import numpy as np
import joblib

# https://towardsdatascience.com/comprehensive-guide-on-multiclass-classification-metrics-af94cfb83fbd
# Compare one classifier’s overall performance to another in a single metric — use Matthew’s correlation coefficient, Cohen’s kappa, and log loss.
# Measure a classifier’s ability to differentiate between each class in balanced classification: ROC AUC score
# A metric that minimizes false positives and false negatives in imbalanced classification: F1 score
# Focus on decreasing the false positives of a single class: Precision for that class
# Focus on decreasing the false negatives of a single class: Recall for that class.

print('Loading test data...')
test_data = np.loadtxt('test_data.csv', delimiter=',')
test_labels = np.loadtxt('test_labels.csv', delimiter=',')
print('Finished loading test data')

print('Loading trained decision tree...')
tree = joblib.load('dtree.joblib')
print('Finished loading trained decision tree')

# print('--- Score (mean accuracy or percentage of correct prediction) ---')
# score = tree.score(test_data, test_labels)
# print(score)

predictions = tree.predict(test_data)

cm = metrics.confusion_matrix(test_labels, predictions)

print('--- Confusion matrix ---')
print(f"""         Predicted NO    |   Predicted YES
Actual NO       {cm[0][0]}              {cm[0][1]}
Actual YES      {cm[1][0]}              {cm[1][1]}""")

f1_score = metrics.f1_score(test_labels, predictions)

print('--- F1-Score ---')
print(f1_score)