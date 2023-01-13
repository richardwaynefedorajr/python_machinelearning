import torch
import torchvision
import torchmetrics

from torch import nn
from torchvision import transforms
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
    ax.set(xlabel='Epochs', ylabel='Metrics', title='Pytorch CNN Training Accuracy and Loss')
    ax.grid()
    ax.legend()
    fig.savefig("pytorch_CNN_metrics.svg")
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
    
def show_images(imgs, num_subplots, titles=None, scale=1.5):
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
            
    fig.suptitle('CNN mis-labeled examples from MNIST Fashion dataset', fontsize=title_fontsize)
    plt.tight_layout()
    plt.savefig('CNN_mislabeled.svg', bbox_inches='tight')
    return axes

def initializeWeights(layer):
    if isinstance(layer, nn.LazyConv2d) or isinstance(layer, nn.LazyLinear):
        torch.nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
                layer.bias.data.zero_()

# hyperparameters
batch_size = 256
output_classes = 10
lr = 0.25
epochs = 10

cm = ConfusionMatrix(task='multiclass', num_classes=output_classes)
training_loss = np.empty(epochs)
training_accuracy = np.empty(epochs)

# activation and loss
activation_function = nn.ReLU()
loss_function = F.cross_entropy

# load data
trans = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
train = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=trans, download=True)
val = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=trans, download=True)

# generate model (flatten layer output dimension is 576)
model = nn.Sequential(
                    nn.Conv2d(1, 6, kernel_size=5, padding=2), 
                    nn.BatchNorm2d(6),
                    activation_function,
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    
                    nn.Conv2d(6, 16, kernel_size=5), 
                    nn.BatchNorm2d(16),
                    activation_function,
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    
                    nn.Flatten(),
                    
                    nn.Linear(576, 120), 
                    nn.BatchNorm1d(120),
                    activation_function,
                    
                    nn.Linear(120, 84), 
                    nn.BatchNorm1d(84),
                    activation_function,
                    
                    nn.Linear(84, output_classes)
                    )

# initialize
model.apply(initializeWeights)
optimizer = torch.optim.SGD(model.parameters(), lr = lr)
accuracy = 0

# fit
for epoch in range(epochs):
    for X, y in iter(torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)):
        y_hat = model(X)
        loss = loss_function(y_hat.reshape((-1, y_hat.shape[-1])), y.reshape((-1,)))
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
plotHeatmap(confusion_matrix_val.numpy(), np.mean(y_hat.numpy() == y.numpy()), ax, 'CNN Validation Results')
plt.savefig('CNN_confusion_matrix.svg', bbox_inches='tight')

# visualize mis-labeled data
wrong = y_hat != y
X, y, y_hat = X[wrong], y[wrong], y_hat[wrong]
labels = [a+'\n'+b for a, b in zip(getLabelsFromIndex(y), getLabelsFromIndex(y_hat))]
show_images(X.squeeze(1), len(y_hat), scale=1.5, titles=labels)

