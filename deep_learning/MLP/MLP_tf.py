import sys
sys.path.append('../utils')
from model import ModelTF

import tensorflow as tf

class MLP(ModelTF):
    def __init__(self, batch_size=256, lr=0.1, epochs=10, input_dimension=None, labels=None):
        super().__init__(batch_size, lr, epochs, labels)

        self.input_dimension = input_dimension

        self.model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                                 tf.keras.layers.Dense(self.input_dimension, activation=self.activation),
                                                 tf.keras.layers.Dense(self.output_dimension)]) 
        
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr)
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
        self.activation = tf.nn.relu
        self.history = None
        
        self.file_prefix = 'MLP'

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