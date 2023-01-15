import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        
    def linePlot(self, data, title):
        sns.set(rc={"figure.dpi":600, 'savefig.dpi':600})
        vals = np.linspace(0,1,256)
        np.random.shuffle(vals)
        cmap = plt.cm.colors.ListedColormap(plt.cm.gist_rainbow(vals))
        sns.set_palette(cmap(np.linspace(0,1,cmap.N)))
        sns.set_style('darkgrid', {'axes.facecolor':'0.85'})
        plot = sns.lineplot(data, dashes=False)
        plot.figure.suptitle(title)
        return plot
    
    def heatmap(self, ax, title, xlabel, ylabel, data):
        sns.set(rc={"figure.dpi":600, 'savefig.dpi':600})
        sns.heatmap(data, annot=True, fmt=".2%", linewidths=.5, square = True, cmap='YlGnBu',
                xticklabels=self.labels, yticklabels=self.labels, ax=ax, cbar_kws={"shrink": 0.75}, annot_kws={'fontsize':30})
        ax.xaxis.set_tick_params(labelsize = 18)
        ax.yaxis.set_tick_params(labelsize = 18)
        ax.set_ylabel(ylabel, fontsize=24)
        ax.set_xlabel(xlabel, fontsize=24)
        ax.set_title(title, fontsize=36)
        return ax
    
    def plotTrainingMetrics(self):
        df = pd.DataFrame(data=np.c_[range(self.epochs), self.training_loss, self.training_accuracy], 
                          columns=['Epochs','Normalized training loss','Training accuracy']).set_index('Epochs')
        plot = self.linePlot(df, self.file_prefix+' Training Metrics')
        plt.savefig(self.file_prefix+'_training_metrics.svg', bbox_inches='tight')
        
    def plotConfusionMatrix(self, y, y_hat):
        self.confusion_matrix = confusion_matrix(y, y_hat, labels=np.arange(0,self.output_dimension))
        fig, ax = plt.subplots(figsize=(30,30))
        accuracy = np.mean(y_hat == y)
        ax = self.heatmap(ax, self.file_prefix+' Confusion Matrix: Accuracy = {}'.format(round(accuracy*100, 2)), 
                          'Predicted Label', 'Actual Label', self.confusion_matrix/self.confusion_matrix.max())
        plt.savefig(self.file_prefix+'_confusion_matrix.svg', bbox_inches='tight')