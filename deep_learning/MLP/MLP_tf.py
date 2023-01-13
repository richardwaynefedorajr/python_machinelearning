import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={"figure.dpi":600, 'savefig.dpi':600})

def plotTrainingMetrics(loss, accuracy, epochs):
    fig, ax = plt.subplots()
    ax.plot(range(epochs), loss/loss.max(), label='Training loss')
    ax.plot(range(epochs), accuracy, label='Training accuracy')
    ax.set(xlabel='Epochs', ylabel='Metrics', title='Tensorflow MLP Training Accuracy and Loss')
    ax.grid()
    ax.legend()
    fig.savefig("tensorflow_MLP_metrics.svg")
    plt.show()

def plotHeatmap(confusion_matrix, accuracy, ax, method):
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True, fmt=".2%", linewidths=.5, square = True, cmap = 'Greens',
            xticklabels=labels, yticklabels=labels, ax=ax, cbar_kws={"shrink": 0.75}, annot_kws={'fontsize':30})
    ax.xaxis.set_tick_params(labelsize = 18)
    ax.yaxis.set_tick_params(labelsize = 18)
    ax.set_ylabel('Actual label', fontsize=24)
    ax.set_xlabel('Predicted label', fontsize=24)
    ax.set_title('{} Accuracy: {}%'.format(method, round(accuracy*100, 2)), fontsize=36)
    
# hyperparameters
batch_size = 256
input_dimension = 784
output_dimension = 10
learning_rate = 0.00015
epochs = 10

# activation and loss
activation = tf.nn.relu
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 

# load data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# create model
model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(input_dimension, activation=activation),
            tf.keras.layers.Dense(output_dimension)]) 

# loss function and optimizer
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), loss=loss_function, metrics=['accuracy'])

# fit
history = model.fit(X_train, y_train, epochs=epochs, verbose=1, batch_size=batch_size, shuffle=True)

# predict
X_test = X_test[:batch_size]
y_test = y_test[:batch_size]
y_hat = model.predict(X_test).argmax(axis=1)

# evaluate
plotTrainingMetrics(np.array(history.history['loss']), np.array(history.history['accuracy']), epochs)
confusion_matrix_val = tf.math.confusion_matrix(y_hat, y_test, num_classes=output_dimension)
fig, ax = plt.subplots(figsize=(30,30))
plotHeatmap(confusion_matrix_val, np.mean(y_hat == y_test), ax, 'MLP Validation Results')
plt.savefig('MLP_confusion_matrix.svg', bbox_inches='tight')