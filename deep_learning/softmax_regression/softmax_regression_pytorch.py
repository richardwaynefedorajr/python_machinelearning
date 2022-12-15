import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
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
            
    fig.suptitle('Softmax regression on MNIST Fashion dataset: Accuracy = {}%'.format(round(accuracy*100,2)), 
                 fontsize=title_fontsize)
    plt.tight_layout()
    plt.savefig('softmax_regression_torch.svg', bbox_inches='tight')
    return axes

def softmax(X):
    return torch.exp(X) / X_exp.sum(1, keepdims=True)

# hyperparameters
batch_size = 256
num_outputs = 10
lr = 0.25
epochs = 10

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

# predict
X, y = next(iter(torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=0)))
y_hat = model(X).argmax(axis=1)

# evaluate
wrong = y_hat != y
X, y, y_hat = X[wrong], y[wrong], y_hat[wrong]
accuracy = 1 - len(y_hat)/batch_size
labels = [a+'\n'+b for a, b in zip(getLabelsFromIndex(y), getLabelsFromIndex(y_hat))]
show_images(X.squeeze(1), len(y_hat), scale=1.5, titles=labels, accuracy=accuracy)