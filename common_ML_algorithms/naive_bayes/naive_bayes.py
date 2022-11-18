from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## load, split, and preprocess data
cancer = datasets.load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split( cancer.data, cancer.target, test_size = 0.1, random_state = 100 )

gnb = GaussianNB().fit(X_train, y_train)
y_hat = gnb.predict(X_test)
print('Accuracy: '+str(round(100 * accuracy_score(y_test, y_hat), 2))+'%')

## dataframe for seaborn plot
cancer_df = pd.DataFrame(np.c_[X_test, y_test], columns= np.append(cancer['feature_names'], ['target']))
cancer_df = cancer_df.sort_values(by=['target'])

## get pairs of feature names for top 5 non-duplicate correlation values (exclude 1 as it is on the diagonal), 
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

## plot correlation heatmap
plt.figure(figsize=(20,12))
sns.heatmap(cancer_df.corr(), annot=True)
plt.title('Feature correlation heatmap')
plt.savefig('naive_bayes_corr_heatmap.png', bbox_inches='tight')
plt.show()