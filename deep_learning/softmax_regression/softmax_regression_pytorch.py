import sys
sys.path.append('../utils')
from model import Model

import torch
import torchvision
import torchmetrics
import numpy as np

from torchvision import transforms
from torch import nn
from torch.nn import functional as F
from torchmetrics import ConfusionMatrix

class Softmax(Model):
    def __init__(self, batch_size=256, lr=0.1, epochs=10, labels=None):
        super().__init__(batch_size, lr, epochs, labels)
        
        self.model = nn.Sequential(nn.Flatten(), nn.LazyLinear(self.output_dimension))
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.loss_function = F.cross_entropy
        
        self.file_prefix = 'softmax'
                
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
    
def softmax(X):
    return torch.exp(X) / X_exp.sum(1, keepdims=True)

# load data
trans = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
train = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=trans, download=True)
val = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=trans, download=True)

softmax = Softmax(batch_size=256, lr=0.0025, epochs=10, 
                  labels=['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'])
softmax.fit(train)
softmax.predict(val)

softmax.plotTrainingMetrics()
softmax.plotConfusionMatrix()