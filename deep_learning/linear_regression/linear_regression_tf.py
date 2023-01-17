import sys
sys.path.append('../utils')
from model import ModelTF

import random
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

class LinearRegression(ModelTF):
    def __init__(self, batch_size=256, lr=0.1, epochs=10, labels=None):
        super().__init__(batch_size, lr, epochs, labels)
        self.output_dimension = 1
        
        self.model = tf.keras.Sequential(layers.Dense(units=self.output_dimension))
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr)
        self.loss_function = 'mean_squared_error'
        
        self.file_prefix = 'tensorflow_linear_regression'

    def predict(self, X, y):
        self.X = X
        self.y = y[:self.batch_size]
        self.y_hat = self.model.predict(X[:self.batch_size])
    
    def plotLinearRegression(self, title):
        super().scatterPlot(title)
        
# generate data
w = tf.constant([-3.2]) #, -3.4])
b = 6.4
noise = 0.5
num_samples = 100
X = tf.random.normal((num_samples, w.shape[0]))
noise = tf.random.normal((num_samples, 1)) * noise
y = tf.matmul(X, tf.reshape(w, (-1, 1))) + b + noise

# create model
model = LinearRegression(batch_size=num_samples, lr=0.1, epochs=250)
model.compile()

# train
model.fit(X, y)

# predict
model.predict(X, y)

# evaluate
model.plotLinearRegression('Tensorflow linear regression')