import sys
sys.path.append('../utils')
from model import Model

import torch
import torchvision
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torchvision import transforms
from torch.nn import functional as F
from torchmetrics import ConfusionMatrix

class CNN(Model):
    def __init__(self, batch_size=256, lr=0.1, epochs=10, labels=None):
        super().__init__(batch_size, lr, epochs, labels)
        
        self.activation = nn.ReLU()

        self.model = nn.Sequential(
                                    nn.Conv2d(1, 6, kernel_size=5, padding=2), 
                                    nn.BatchNorm2d(6),
                                    self.activation,
                                    nn.AvgPool2d(kernel_size=2, stride=2),
                                    
                                    nn.Conv2d(6, 16, kernel_size=5), 
                                    nn.BatchNorm2d(16),
                                    self.activation,
                                    nn.AvgPool2d(kernel_size=2, stride=2),
                                    
                                    nn.Flatten(),
                                    
                                    nn.Linear(576, 120), 
                                    nn.BatchNorm1d(120),
                                    self.activation,
                                    
                                    nn.Linear(120, 84), 
                                    nn.BatchNorm1d(84),
                                    self.activation,
                                    
                                    nn.Linear(84, self.output_dimension)
                                )
        
        self.loss_function = F.cross_entropy
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = lr)
        
        self.file_prefix = 'CNN'
    
    def initWeights(self):
        def initializeWeights(layer):
            if isinstance(layer, nn.LazyConv2d) or isinstance(layer, nn.LazyLinear):
                torch.nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                        layer.bias.data.zero_()
                        
        self.model.apply(initializeWeights)
        
    def fit(self, data):
        for epoch in range(self.epochs):
            for self.X, self.y in iter(torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=0)):
                self.y_hat = self.model(self.X)
                loss = self.loss_function(self.y_hat.reshape((-1, self.y_hat.shape[-1])), self.y.reshape((-1,)), reduction='mean')
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print('epoch {}, loss {}'.format(epoch, loss.item()))
            self.training_loss[epoch] = loss.item()
            self.training_accuracy[epoch] = np.mean(self.y_hat.argmax(axis=1).numpy() == self.y.numpy())
        
    def predict(self, data):
        self.X, self.y = next(iter(torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=0)))
        self.y_hat = self.model(self.X).argmax(axis=1)
        
    def plotConfusionMatrix(self):
        super().plotConfusionMatrix(self.y.numpy(), self.y_hat.numpy())
            
    def getLabelsFromIndex(self, labels):
        return [self.labels[int(i)] for i in labels]
    
    def setSubplotDimensions(self, num_subplots):
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
        
    def show_images(self, imgs, num_subplots, titles=None, scale=2):
        label_fontsize, title_fontsize, colormap = 18, 24, 'viridis'
        num_rows, num_cols = self.setSubplotDimensions(num_subplots)
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
                
        fig.suptitle(self.file_prefix+' mis-labeled examples from MNIST Fashion dataset', fontsize=title_fontsize)
        plt.tight_layout()
        plt.savefig(self.file_prefix+'_mislabeled.svg', bbox_inches='tight')
        return axes

    def plotIncorrectExamples(self):
        wrong = self.y_hat != self.y
        X, y, y_hat = self.X[wrong], self.y[wrong], self.y_hat[wrong]
        example_labels = [a+'\n'+b for a, b in zip(self.getLabelsFromIndex(y), self.getLabelsFromIndex(y_hat))]
        self.show_images(X.squeeze(1), len(y_hat), scale=1.5, titles=example_labels)
        
# load data
trans = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
train = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=trans, download=True)
val = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=trans, download=True)

cnn = CNN(batch_size=256, lr=0.25, epochs=10, 
                  labels=['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'])
cnn.initWeights()
cnn.fit(train)
cnn.predict(val)

cnn.plotTrainingMetrics()
cnn.plotConfusionMatrix()
cnn.plotIncorrectExamples()