import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

## import data
cancer = datasets.load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data,
                                                    cancer.target,
                                                    test_size=0.3,
                                                    random_state=109)

## train
clf = svm.SVC(kernel='linear').fit(X_train, y_train)

## predict
y_hat = clf.predict(X_test)

# accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_hat))
print("Precision:",metrics.precision_score(y_test, y_hat))
print("Recall:",metrics.recall_score(y_test, y_hat))