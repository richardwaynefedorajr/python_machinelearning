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

## import data
wine = datasets.load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=109)

## train
sk_svm = svm.SVC(kernel='linear').fit(X_train, y_train)

## predict
y_hat = sk_svm.predict(X_test)

## evaluate
print('Accuracy: '+str(round(100*metrics.accuracy_score(y_test, y_hat), 2))+'%')
# print('Precision: '+str(round(100*metrics.precision_score(y_test, y_hat, average='weighted'), 2))+'%')
# print('Recall: '+str(round(100*metrics.recall_score(y_test, y_hat, average='weighted'), 2))+'%')
print('\n')

## normalize data
sk_svm_normalized = Pipeline([('scaler', StandardScaler()), ('svm_clf', svm.SVC(kernel='linear'))]).fit(X_train, y_train)
y_hat_normalized = sk_svm_normalized.predict(X_test)
print('Accuracy: '+str(round(100*metrics.accuracy_score(y_test, y_hat_normalized), 2))+'%')
# print('Precision: '+str(round(100*metrics.precision_score(y_test, y_hat_normalized, average='weighted'), 2))+'%')
# print('Recall: '+str(round(100*metrics.recall_score(y_test, y_hat_normalized, average='weighted'), 2))+'%')

## add data to plot
X_test = normalize(X_test, axis=0, norm='max')
y_hat_normalized[y_hat_normalized != y_test] = 3.0
wine_df = pd.DataFrame(data=np.c_[X_test, y_hat_normalized], columns=wine['feature_names'] + ['target'])
df_plot = pd.DataFrame(data=np.c_[wine_df[wine.feature_names].mean(axis=1), wine_df[wine.feature_names].std(axis=1), wine_df['target']], 
                       columns=['Normalized feature vector mean','Normalized feature vector standard deviation', 'target'])
df_plot['target'][df_plot["target"] == 0.0] = wine.target_names[0]
df_plot['target'][df_plot["target"] == 1.0] = wine.target_names[1]
df_plot['target'][df_plot["target"] == 2.0] = wine.target_names[2]
df_plot['target'][df_plot["target"] == 3.0] = 'Misclassified'

sns.scatterplot(data=df_plot, x='Normalized feature vector mean', y='Normalized feature vector standard deviation', 
                hue="target", style='target', palette=['greenyellow','lime', 'darkgreen', 'red'], markers=['D','D','D','X'])

plt.suptitle('SVM (normalized data) accuracy: '+str(round(metrics.accuracy_score(y_test, y_hat_normalized)*100, 2))+'%', y=1.025)
plt.savefig('svm_scatter_predicted.png', bbox_inches='tight')
plt.show()