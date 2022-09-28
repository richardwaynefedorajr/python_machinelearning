import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import math

## load and split
iris_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split( iris_data.data,
                                                     iris_data.target,
                                                     test_size = 0.3,
                                                     random_state = 100 )
y_train = y_train.reshape((-1,1))
X_train = np.concatenate([X_train, np.ones_like(y_train, dtype=np.float32)], axis=1)
X_test = np.concatenate([X_test, np.ones((X_test.shape[0], 1), dtype=np.float32)], axis=1)

## train
if X_train.shape[0] >= X_train.shape[1] == np.linalg.matrix_rank(X_train):
    y_train = np.maximum(1e-5, np.minimum(y_train.astype(np.float32), 1-1e-5))
    # w = [(X^T*T)^-1*X^T]*log(1/y) -1
    weights =  np.matmul( np.matmul( np.linalg.inv( np.matmul( X_train.transpose(), X_train) ),
               X_train.transpose()), -np.log(np.divide(1, y_train) - 1))
else:
    print('X does not have full column rank')
    weights = 0

## predict
y_hat = np.divide(1, 1+np.exp(-np.matmul(X_test, weights)))

## accuracy
print('Accuracy')
zeros, ones = np.zeros_like(y_hat), np.ones_like(y_hat)
y_test = np.where(y_test >= 0.5, ones, zeros)
y_hat = np.where(y_hat >= 0.5, ones, zeros)
print(np.mean((y_test == y_hat).astype(np.float32)))
