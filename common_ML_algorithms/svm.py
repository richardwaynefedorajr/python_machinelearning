import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import tensorflow as tf

## import data
cancer = datasets.load_breast_cancer()
cancer_df = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns= np.append(cancer['feature_names'], ['target']))
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=109)

## train
sk_svm = svm.SVC(kernel='linear').fit(X_train, y_train)

## predict
y_hat = clf.predict(X_test)

## accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_hat))
print("Precision:",metrics.precision_score(y_test, y_hat))
print("Recall:",metrics.recall_score(y_test, y_hat))

## plotting
#sns.pairplot(cancer_df, hue='target', vars=cancer.feature_names[0:3])
#plt.figure(figsize=(20,12))
#sns.heatmap(cancer_df.corr(), annot=True)