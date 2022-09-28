import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
import tensorflow as tf
import math

## linear regression numpy implementation

def preprocess():
    ## generate test data
    x, y = make_regression(n_features=1, n_informative=1, bias=1, noise=35)

    ## make y a column vector and add bias column to x
    y = y.reshape((-1, 1))
    x = np.concatenate([x, np.ones_like(y)], axis=1)
    return x, y

def train_lr(x, y, method_and_library):        
    if method_and_library == 'numpy_solve':
        if x.shape[0] >= x.shape[1] == np.linalg.matrix_rank(x):
            print('Numpy solve method')
            return np.matmul( np.matmul( np.linalg.inv( np.matmul( x.transpose(), x)), x.transpose()), y)
        else:
            print('X missing full column rank: return 0 weights')
            return np.zeros((cols, 1))
    elif method_and_library == 'tf_sgd':
        print('TensorFlow stochastic gradient descent')
    elif method_and_library == 'sk_regression':
        print('Scikit-learn regression:')
    else:
        print('No method returned for '+method_and_library+': return 0 weights')
        return np.zeros((cols, 1))
            
## preprocess data and add to plot
x, y = preprocess()
plt.scatter(x[:,:-1], y)

## "train" -> calculate weights for linear regression best fit
weights = train_lr(x, y, 'numpy_solve')

y_hat = np.matmul(x, weights)

plt.plot(x[:,:-1], y_hat, color='orange')
plt.show()
