import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math

## linear regression numpy implementation

def preprocess():
    ## generate test data
    x, y = make_regression(n_features=1, n_informative=1, bias=1, noise=35)

    ## make y a column vector and add bias column to x
    y = y.reshape((-1, 1))
    x = np.concatenate([x, np.ones_like(y)], axis=1)
    return x, y

## preprocess data and add to plot
x, y = preprocess()
plt.scatter(x[:,:-1], y)

## "train" -> calculate weights for linear regression best fit
## numpy closed form implementation
if x.shape[0] >= x.shape[1] == np.linalg.matrix_rank(x):
    weights = np.matmul( np.matmul( np.linalg.inv( np.matmul( x.transpose(), x)), x.transpose()), y)
else:
    print('X missing full column rank: return 0 weights')
    weights = np.zeros((cols, 1))

## tensorflow implementation
tf_model = tf.keras.Sequential(layers.Dense(units=1))
tf_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),loss='mean_absolute_error')
tf_model.fit(x,y,epochs=100,verbose=0,validation_split = 0.2)
tf_model.summary()

## predict
y_hat_solve = np.matmul(x, weights)
y_hat_tf = tf_model.predict(x)

plt.plot(x[:,:-1], y_hat_solve, color='blue')
plt.plot(x[:,:-1], y_hat_tf, color='orange')
plt.show()
