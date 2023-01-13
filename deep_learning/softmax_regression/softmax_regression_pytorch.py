import torch
import torchvision
import torchmetrics

from torchvision import transforms
from torch import nn
from torch.nn import functional as F
from torchmetrics import ConfusionMatrix

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={"figure.dpi":600, 'savefig.dpi':600})

def plotTrainingMetrics(loss, accuracy, epochs):
    fig, ax = plt.subplots()
    ax.plot(range(epochs), loss/loss.max(), label='Training loss')
    ax.plot(range(epochs), accuracy, label='Training accuracy')
    ax.set(xlabel='Epochs', ylabel='Metrics', title='Pytorch Softmax Training Accuracy and Loss')
    ax.grid()
    ax.legend()
    fig.savefig("pytorch_softmax_metrics.svg")
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
    
def softmax(X):
    return torch.exp(X) / X_exp.sum(1, keepdims=True)

# hyperparameters
batch_size = 256
num_outputs = 10
lr = 0.0025
epochs = 10

cm = ConfusionMatrix(task='multiclass', num_classes=num_outputs)
training_loss = np.empty(epochs)
training_accuracy = np.empty(epochs)

# load data
trans = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
train = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=trans, download=True)
val = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=trans, download=True)

# generate model
model = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_outputs))
optimizer = torch.optim.SGD(model.parameters(), lr = lr)

# fit
for epoch in range(epochs):
    for X, y in iter(torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)):
        y_hat = model(X)
        loss = F.cross_entropy(y_hat.reshape((-1, y_hat.shape[-1])), y.reshape((-1,)), reduction='mean')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.item()))
    training_loss[epoch] = loss.item()
    training_accuracy[epoch] = np.mean(y_hat.argmax(axis=1).numpy() == y.numpy())

# predict
X, y = next(iter(torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=0)))
y_hat = model(X).argmax(axis=1)

# evaluate
plotTrainingMetrics(training_loss, training_accuracy, epochs)
confusion_matrix_val = cm(y_hat, y)
fig, ax = plt.subplots(figsize=(30,30))
plotHeatmap(confusion_matrix_val.numpy(), np.mean(y_hat.numpy() == y.numpy()), ax, 'Softmax Regression Validation Results')
plt.savefig('softmax_confusion_matrix.svg', bbox_inches='tight')