import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

sns.set(rc={"figure.dpi":600, 'savefig.dpi':600})

## load, split, and preprocess data
cancer = datasets.load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split( cancer.data, cancer.target, test_size = 0.1, random_state = 100 )

# massage data for numpy solution
y_train_np = y_train.reshape((-1,1))
y_test_np = y_test.reshape((-1,1))
X_train_np = np.concatenate([X_train, np.ones_like(y_train_np, dtype=np.float32)], axis=1)
X_test_np = np.concatenate([X_test, np.ones((X_test.shape[0], 1), dtype=np.float32)], axis=1)

## train
sk_model = LogisticRegression().fit(X_train, y_train)

if X_train_np.shape[0] >= X_train_np.shape[1] == np.linalg.matrix_rank(X_train_np):
    y_train_np = np.maximum(1e-5, np.minimum(y_train_np.astype(np.float32), 1-1e-5))
    # w = [(X^T*T)^-1*X^T]*log(1/y) -1
    weights =  np.matmul( np.matmul( np.linalg.inv( np.matmul( X_train_np.transpose(), X_train_np) ),
               X_train_np.transpose()), -np.log(np.divide(1, y_train_np) - 1))
else:
    print('X does not have full column rank')
    weights = 0
    
## predict
y_hat_sk = sk_model.predict(X_test)
y_hat_np = np.divide(1, 1+np.exp(-np.matmul(X_test_np, weights)))
zeros, ones = np.zeros_like(y_hat_np), np.ones_like(y_hat_np)
y_hat_np = np.where(y_hat_np >= 0.5, ones, zeros)

## evaluate
confusion_matrix_sk = metrics.confusion_matrix(y_test, y_hat_sk)
confusion_matrix_np = metrics.confusion_matrix(y_test, y_hat_np)

## plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(40,20))
def plotHeatmap(confusion_matrix, y_hat, y_test, ax, method):
    sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True, fmt=".2%", linewidths=.5, square = True, 
            cmap = 'Greens', xticklabels=['malignant','benign'], yticklabels=['malignant','benign'], 
            ax=ax, cbar_kws={"shrink": 0.75}, annot_kws={'fontsize':30})
    ax.xaxis.set_tick_params(labelsize = 18)
    ax.yaxis.set_tick_params(labelsize = 18)
    ax.set_ylabel('Actual label', fontsize=24)
    ax.set_xlabel('Predicted label', fontsize=24)
    ax.set_title('{} Accuracy: {}'.format(method, round(np.mean(y_hat == y_test)*100, 2)), fontsize=36)

plotHeatmap(confusion_matrix_sk, y_hat_sk, y_test, ax1, 'Scikit')
plotHeatmap(confusion_matrix_np, y_hat_np, y_test_np, ax2, 'Numpy')
plt.savefig('logistic_regression_confusion_matrices.png', bbox_inches='tight')