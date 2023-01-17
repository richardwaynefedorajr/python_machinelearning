import sys
sys.path.append('../utils')
from model import ModelTF

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

class LinearRegressionSeq(ModelTF):
    def __init__(self, batch_size=256, lr=0.1, epochs=10, labels=None, time_steps=10, tau=1):
        super().__init__(batch_size, lr, epochs, labels)
        self.output_dimension = 1
        self.time_steps = time_steps
        self.tau = tau
        
        self.model = tf.keras.Sequential(layers.Dense(units=self.output_dimension))
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr)
        self.loss_function = 'mean_squared_error'
        
        self.file_prefix = 'sequential_linear_regression'

    def predict(self, features, labels, x):
        self.X = x
        self.y = labels
        self.y_hat = self.model.predict(features)
    
    def plotLinearRegression(self):
        df = pd.DataFrame(data=np.c_[range(self.tau, self.time_steps), self.X[self.tau:], self.y_hat], columns=['Time','X','x_hat']).set_index('Time')
        plot = self.linePlot(df, 'Sequential Linear Regression Results', linewidth=0.25)
        plt.savefig(self.file_prefix+'_results.svg', bbox_inches='tight')
        
# generate synthetic data
time_steps = 1000
num_train=600
tau=4
time = tf.range(1, time_steps + 1, dtype=tf.float32)
x = tf.cos(0.01 * time) + tf.random.normal([time_steps]) * 0.2

features = [x[i : time_steps-tau+i] for i in range(tau)]
features = tf.stack(features, 1)
labels = tf.reshape(x[tau:], (-1, 1))

# create model
model = LinearRegressionSeq(batch_size=16, lr=0.01, epochs=10, time_steps=time_steps, tau=tau)
model.compile()

# fit
model.fit(features, labels)

# predict
model.predict(features, labels, x)

# evaluate
model.plotLinearRegression()