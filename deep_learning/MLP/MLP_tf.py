import sys
sys.path.append('../utils')
from model import Model

import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import np_utils
from keras.utils import plot_model

class MLP(Model):
    def __init__(self, batch_size=256, lr=0.1, epochs=10, input_dimension=None, labels=None):
        super().__init__(batch_size, lr, epochs, labels)

        self.input_dimension = input_dimension

        self.model = tf.keras.models.Sequential([
                                                 tf.keras.layers.Flatten(),
                                                 tf.keras.layers.Dense(self.input_dimension, activation=self.activation),
                                                 tf.keras.layers.Dense(self.output_dimension)]) 
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr)
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
        self.activation = tf.nn.relu
        self.history = None
        
        self.file_prefix = 'MLP'

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=['accuracy'])
        
    def fit(self, X, y):
        self.history = self.model.fit(X, y, epochs=self.epochs, verbose=1, batch_size=self.batch_size, shuffle=True)
        self.training_accuracy = np.array(self.history.history['accuracy'])
        self.training_loss = np.array(self.history.history['loss'])
        
    def predict(self, X, y):
        self.y = y[:self.batch_size]
        self.y_hat = self.model.predict(X[:self.batch_size]).argmax(axis=1)

    def plotConfusionMatrix(self):
        super().plotConfusionMatrix(self.y, self.y_hat)
        
    def showNetworkStructure(self):
        plot_model(self.model, to_file=self.file_prefix+'_structure.png', show_shapes=True)
        
# load data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# parameterize model, train, and predict
model = MLP(batch_size=256, lr=0.00015, epochs=10, input_dimension=784,
            labels=['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'])   
model.compile()     
model.fit(X_train, y_train)
model.predict(X_test, y_test)

# evaluate
model.plotTrainingMetrics()
model.plotConfusionMatrix()
# model.showNetworkStructure()