import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
sns.set(rc={"figure.dpi":600, 'savefig.dpi':600})

# hyperparameters
w = tf.constant([2.0]) #, -3.4])
b = 4.2
noise = 0.5
num_samples = 100

# generate data
X = tf.random.normal((num_samples, w.shape[0]))
noise = tf.random.normal((num_samples, 1)) * noise
y = tf.matmul(X, tf.reshape(w, (-1, 1))) + b + noise

# create model
tf_model = tf.keras.Sequential(layers.Dense(units=1))

# loss function and optimizer
tf_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), loss='mean_squared_error')

# fit
tf_model.fit(X, y, epochs=250, verbose=0, validation_split=0.2)

# predict
y_hat = tf_model.predict(X)

# evaluate
squared_error = np.square(y - y_hat)

## add data to plot
fig = plt.figure()
ax = plt.axes()
df = pd.DataFrame(data=np.c_[X[:,0], y, y_hat], columns=['X', 'y', 'y_hat'])
sns.scatterplot(data=df, x='X', y='y', ax=ax, edgecolor='black', size=np.abs(y-y_hat)[:,0], c='greenyellow')
# sns.scatterplot(data=df, x='X', y='y_hat', ax=ax, marker='o', edgecolor='black', s=2, c='darkgreen')
ax.plot(X[:,0], y_hat, color='green', linestyle='solid', linewidth=1, label='y_hat')
ax.set_title('Tensorflow linear regression [MSE = '+str(round(squared_error.mean(),2))+']')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(fontsize=8, title_fontsize=8, title='marker size for\nerror values')
plt.savefig('linear_regression_tensorflow.svg', bbox_inches='tight')
plt.show()