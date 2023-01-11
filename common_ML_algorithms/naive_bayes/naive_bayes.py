from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={"figure.dpi":600, 'savefig.dpi':600})

## load, split, and preprocess data
cancer = datasets.load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split( cancer.data, cancer.target, test_size = 0.1, random_state = 100 )

gnb = GaussianNB().fit(X_train, y_train)
y_hat = gnb.predict(X_test)
print('Accuracy: '+str(round(100 * accuracy_score(y_test, y_hat), 2))+'%')

## dataframe for seaborn plot
cancer_df = pd.DataFrame(np.c_[X_test, y_test], columns= np.append(cancer['feature_names'], ['target']))
cancer_df = cancer_df.sort_values(by=['target'])

## evaluate
confusion_matrix_full = confusion_matrix(y_test, y_hat)

## get pairs of feature names for top non-duplicate correlation values (exclude 1 as it is on the diagonal), 
## get unique names using set, and convert to a list (itertools.chain converts list of tuples of feature pairs into single iterator)
max_corr = list(set(itertools.chain(*cancer_df.corr().abs().unstack().sort_values(ascending=False).drop_duplicates()
                                   .iloc[1:3].index.values)))
cols_to_drop = cancer_df.columns.get_indexer(max_corr)
X_train = np.delete(X_train, cols_to_drop, 1)
X_test = np.delete(X_test, cols_to_drop, 1)
gnb_corr = GaussianNB().fit(X_train, y_train)
y_hat_corr = gnb_corr.predict(X_test)
print('Accuracy after removing '+str(len(cols_to_drop))+' most highly correlated features: '
      +str(round(100 * accuracy_score(y_test, y_hat_corr), 2))+'%')
print('Dropped features: '+', '.join(max_corr))

## evaluate
confusion_matrix_dropped_features = confusion_matrix(y_test, y_hat_corr)

## plot correlation heatmap
plt.figure(figsize=(20,12))
sns.heatmap(cancer_df.corr(), annot=True)
plt.title('Feature correlation heatmap', fontsize=18)
plt.savefig('naive_bayes_corr_heatmap.svg', bbox_inches='tight')
plt.show()

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

plotHeatmap(confusion_matrix_full, y_hat, y_test, ax1, 'Full Feature')
plotHeatmap(confusion_matrix_dropped_features, y_hat_corr, y_test, ax2, 'Dimensionality Reduction')
plt.savefig('naive_bayes_confusion_matrices.svg', bbox_inches='tight')