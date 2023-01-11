import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.pipeline import Pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={"figure.dpi":600, 'savefig.dpi':600})

## import data
cancer = datasets.load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=109)

## train
sk_svm = svm.SVC(kernel='linear').fit(X_train, y_train)

## predict
y_hat = sk_svm.predict(X_test)

## evaluate
print('Accuracy: '+str(round(100*metrics.accuracy_score(y_test, y_hat), 2))+'%')
# print('Precision: '+str(round(100*metrics.precision_score(y_test, y_hat, average='weighted'), 2))+'%')
# print('Recall: '+str(round(100*metrics.recall_score(y_test, y_hat, average='weighted'), 2))+'%')
print('\n')
confusion_matrix_raw = metrics.confusion_matrix(y_test, y_hat)

## normalize data
sk_svm_normalized = Pipeline([('scaler', StandardScaler()), ('svm_clf', svm.SVC(kernel='linear'))]).fit(X_train, y_train)
y_hat_normalized = sk_svm_normalized.predict(X_test)
print('Accuracy: '+str(round(100*metrics.accuracy_score(y_test, y_hat_normalized), 2))+'%')
# print('Precision: '+str(round(100*metrics.precision_score(y_test, y_hat_normalized, average='weighted'), 2))+'%')
# print('Recall: '+str(round(100*metrics.recall_score(y_test, y_hat_normalized, average='weighted'), 2))+'%')

## evaluate
confusion_matrix_normalized = metrics.confusion_matrix(y_test, y_hat_normalized)

## plot confusion matrix
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(40,20))
def plotHeatmap(confusion_matrix, y_hat, y_test, ax, method):
    sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True, fmt=".2%", linewidths=.5, square = True, 
            cmap = 'Greens', xticklabels=['malignant','benign'], yticklabels=['malignant','benign'], 
            ax=ax, cbar_kws={"shrink": 0.75}, annot_kws={'fontsize':30})
    ax.xaxis.set_tick_params(labelsize = 18)
    ax.yaxis.set_tick_params(labelsize = 18)
    ax.set_ylabel('Actual label', fontsize=24)
    ax.set_xlabel('Predicted label', fontsize=24)
    ax.set_title('{} Accuracy: {}%'.format(method, round(np.mean(y_hat == y_test)*100, 2)), fontsize=36)

plotHeatmap(confusion_matrix_raw, y_hat, y_test, ax1, 'Raw Data')
plotHeatmap(confusion_matrix_normalized, y_hat_normalized, y_test, ax2, 'Normalized Data')
plt.savefig('svm_confusion_matrices.svg', bbox_inches='tight')