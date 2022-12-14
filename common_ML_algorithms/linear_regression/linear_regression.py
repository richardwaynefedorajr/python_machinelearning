import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')
from sklearn.datasets import make_regression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.linear_model import Ridge
from sklearn import linear_model
import pandas as pd
import seaborn as sns
import math
sns.set(rc={"figure.dpi":600, 'savefig.dpi':600})

## generate and preprocess data
x, y = make_regression(n_features=1, n_informative=1, bias=1, noise=35)

## make y a column vector and add bias column to x
y = y.reshape((-1, 1))
x = np.concatenate([x, np.ones_like(y)], axis=1)

## "train" -> calculate weights for linear regression best fit
## numpy closed form implementation
if x.shape[0] >= x.shape[1] == np.linalg.matrix_rank(x):
    weights = np.matmul( np.matmul( np.linalg.inv( np.matmul( x.transpose(), x)), x.transpose()), y)
else:
    print('X missing full column rank: return 0 weights')
    weights = np.zeros((cols, 1))

## tensorflow implementation
tf_model = tf.keras.Sequential(layers.Dense(units=1))
tf_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),loss='mean_squared_error')
tf_model.fit(x,y,epochs=100,verbose=0,validation_split = 0.2)
tf_model.summary()

## ridge regression implementation
sk_ridge = Ridge(alpha=5.0).fit(x,y)


## LASSO implementation
sk_lasso = linear_model.Lasso(alpha=5.0).fit(x,y)

## predict
y_hat_solve = np.matmul(x, weights)
y_hat_tf = tf_model.predict(x)
y_hat_ridge = sk_ridge.predict(x)
y_hat_lasso = sk_lasso.predict(x)

## evaluate
squared_error_solve = np.square(y - y_hat_solve)
squared_error_tf = np.square(y - y_hat_tf)
squared_error_ridge = np.square(y - y_hat_ridge)
squared_error_lasso = np.square(y - y_hat_lasso)

## add data to plot
fig = plt.figure()
ax = plt.axes()
df = pd.DataFrame(data=np.c_[x[:,:-1], y, y_hat_solve, y_hat_tf, squared_error_solve, squared_error_tf], 
                  columns=['x','y','y_hat_solve','y_hat_tf','squared_error_solve','squared_error_tf'])
sns.scatterplot(data=df, x='x', y='y', ax=ax, edgecolor='black', s=20, c='greenyellow')
ax.plot(x[:,:-1], y_hat_solve, color='green', linestyle='solid', linewidth=0.5,
        label='NumPy [MSE = '+str(round(squared_error_solve.mean(),2))+']')
ax.plot(x[:,:-1], y_hat_tf, color='blue', linestyle='solid', linewidth=0.5,
        label='Tensorflow [MSE = '+str(round(squared_error_tf.mean(),2))+']')
ax.plot(x[:,:-1], y_hat_ridge, color='cyan', linestyle='solid', linewidth=0.5,
        label='Scikit Ridge [MSE = '+str(round(squared_error_ridge.mean(),2))+']')
ax.plot(x[:,:-1], y_hat_lasso, color='yellow', linestyle='solid', linewidth=0.5,
        label='Scikit LASSO [MSE = '+str(round(squared_error_lasso.mean(),2))+']')
ax.set_title('Comparison of linear regressions')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(fontsize=8)
plt.savefig('linear_regression.png', bbox_inches='tight')
plt.show()