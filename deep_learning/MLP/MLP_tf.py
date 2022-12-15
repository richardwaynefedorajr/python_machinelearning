import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt

def getLabelsFromIndex(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def setSubplotDimensions(num_subplots):
    min_cols = 6
    max_cols = 10
    possible_num_cols = np.arange(min_cols,max_cols+1)
    num_cols = possible_num_cols[np.argmin(num_subplots % possible_num_cols)]
    if num_subplots < max_cols:
        return 1, num_subplots
    elif num_subplots % num_cols == 0:
        return num_subplots // num_cols, num_cols
    else:
        num_cols = possible_num_cols[np.argmax(num_subplots % possible_num_cols)]
        return num_subplots // num_cols + 1, num_cols
    
def show_images(imgs, num_subplots, titles=None, scale=1.5, accuracy=0.0):
    label_fontsize, title_fontsize, colormap = 18, 24, 'viridis'
    num_rows, num_cols = setSubplotDimensions(num_subplots)
    figsize = (num_cols * scale, num_rows * scale**2)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for ax in axes:
        ax.set_axis_off()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        try:
            img = img.detach().numpy()
        except:
            pass
        ax.imshow(img, cmap=colormap)

        if titles:
            ax.set_title(titles[i], fontsize=label_fontsize)
            
    fig.suptitle('MLP on MNIST Fashion dataset: Accuracy = {}%'.format(round(accuracy*100,2)), 
                 fontsize=title_fontsize)
    plt.tight_layout()
    plt.savefig('MLP_tf.svg', bbox_inches='tight')
    return axes

# hyperparameters
batch_size = 256
num_inputs = 784
num_outputs = 10
num_hiddens = 256
activation = 'relu'
loss_function = 'sparse_categorical_crossentropy'
learning_rate = 0.001
epochs = 20

# load data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# create model
model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_inputs, activation=activation),
            tf.keras.layers.Dense(num_hiddens, activation=activation),
            tf.keras.layers.Dense(num_outputs, activation='softmax')])

# loss function and optimizer
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), loss=loss_function)

# fit
model.fit(X_train, y_train, epochs=epochs, verbose=1, batch_size=batch_size, shuffle=True)

# predict
X_test = X_test[:batch_size]
y_test = y_test[:batch_size]
y_hat = model.predict(X_test).argmax(axis=1)

# evaluate
wrong = y_hat != y_test
X_test, y_test, y_hat = X_test[wrong], y_test[wrong], y_hat[wrong]
accuracy = 1 - len(y_hat)/len(wrong)
labels = [a+'\n'+b for a, b in zip(getLabelsFromIndex(y_test), getLabelsFromIndex(y_hat))]
show_images(X_test, len(y_hat), scale=1.5, titles=labels, accuracy=accuracy)