import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import np_utils
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix

class Model():
    def __init__(self, batch_size=256, lr=0.1, epochs=10, labels=None):
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.training_loss = np.empty(epochs)
        self.training_accuracy = np.empty(epochs)
        self.labels = labels
        self.output_dimension = len(self.labels) if labels else 0
        self.confusion_matrix = np.empty([self.output_dimension,self.output_dimension])
        
        self.X = None
        self.y = None
        self.y_hat = None
        
        self.model = None
        self.optimizer = None
        self.loss_function = None
        self.activation = None
        
        self.file_prefix  = None
        
    def linePlot(self, data, title, linewidth=0.75):
        sns.set(rc={"figure.dpi":600, 'savefig.dpi':600})
        vals = np.linspace(0,1,256)
        np.random.shuffle(vals)
        cmap = plt.cm.colors.ListedColormap(plt.cm.gist_rainbow(vals))
        sns.set_palette(cmap(np.linspace(0,1,cmap.N)))
        sns.set_style('darkgrid', {'axes.facecolor':'0.85'})
        plot = sns.lineplot(data, dashes=False, linewidth=linewidth)
        plot.figure.suptitle(title)
        return plot
    
    def scatterPlot(self, title):
        fig = plt.figure()
        ax = plt.axes()
        df = pd.DataFrame(data=np.c_[self.X[:,0], self.y, self.y_hat], columns=['X', 'y', 'y_hat'])
        accuracy = round(np.mean(np.square(self.y_hat - self.y)), 2)
        sns.scatterplot(data=df, x='X', y='y', ax=ax, edgecolor='black', size=np.abs(self.y-self.y_hat)[:,0], c='greenyellow')
        ax.plot(self.X[:,0], self.y_hat, color='green', linestyle='solid', linewidth=1, label='y_hat')
        ax.set_title(title+' [MSE = '+str(accuracy)+']')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=8, title_fontsize=8, title='marker size for\nerror values')
        plt.savefig(self.file_prefix+'_scatterplot.svg', bbox_inches='tight')
        plt.show()
    
    def heatmap(self, ax, title, data):
        sns.set(rc={"figure.dpi":600, 'savefig.dpi':600})
        sns.heatmap(data, annot=True, fmt=".1%", linewidths=0.5, square=True, cmap='YlGnBu',
                xticklabels=self.labels, yticklabels=self.labels, ax=ax, cbar_kws={"shrink": 0.75}, annot_kws={'fontsize':10})
        ax.xaxis.set_tick_params(labelsize = 10)
        ax.yaxis.set_tick_params(labelsize = 10)
        ax.set_title(title, fontsize=18)
        ax.set_xlabel('Predicted Label', fontsize=18, horizontalalignment='center')
        ax.set_ylabel('Actual Label', fontsize=18, verticalalignment='center')
        return ax
    
    def plotTrainingMetrics(self):
        df = pd.DataFrame(data=np.c_[range(self.epochs), self.training_loss/self.training_loss.max(), self.training_accuracy], 
                          columns=['Epochs','Normalized training loss','Training accuracy']).set_index('Epochs')
        plot = self.linePlot(df, self.file_prefix+' Training Metrics')
        plt.savefig(self.file_prefix+'_training_metrics.svg', bbox_inches='tight')
        
    def plotConfusionMatrix(self, y, y_hat):
        self.confusion_matrix = confusion_matrix(y, y_hat, labels=np.arange(0,self.output_dimension))
        fig, ax = plt.subplots(figsize=(9,8))
        accuracy = np.mean(y_hat == y)
        title = self.file_prefix+' Confusion Matrix: Accuracy = {}'.format(round(accuracy*100, 2))
        ax = self.heatmap(ax, title, self.confusion_matrix/self.confusion_matrix.max())
        plt.savefig(self.file_prefix+'_confusion_matrix.svg', bbox_inches='tight')
        
class ModelTF(Model):
    def __init__(self, batch_size=256, lr=0.1, epochs=10, labels=None):
        super().__init__(batch_size, lr, epochs, labels)
        
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

class ModelTorch(Model):
    pass