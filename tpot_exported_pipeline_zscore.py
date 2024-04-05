import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer

from sklearn.metrics import roc_curve
from sklearn import metrics
import matplotlib.pyplot as plt 


# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('zscore_tpot_data.csv', sep=',', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)
print('here1')
imputer = SimpleImputer(strategy="median")
imputer.fit(training_features)
training_features = imputer.transform(training_features)
testing_features = imputer.transform(testing_features)
print('here2')
# Average CV score on the training set was: 0.8672151898734178
exported_pipeline = XGBClassifier(learning_rate=0.1, max_depth=9, min_child_weight=7, n_estimators=100, n_jobs=1, subsample=0.9000000000000001, verbosity=0)
# Fix random state in exported estimator
print('here3')
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

c = 0
for i in range(len(results)):
    if results[i] == testing_target.array[i]:
        c = c+1

testing_accuracy = c/len(results)*100
print(testing_accuracy)
print("real", testing_target.array)
print("predict", results)


fpr, tpr, thresholds = roc_curve(testing_target.array, results)

auc_val = metrics.auc(fpr, tpr)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='TPOT(area = {:.2f})'.format(auc_val))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
# Zoom in view of the upper left corner.
'''
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='TensorFlow (area = {:.2f})'.format(auc_val))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show()

'''

