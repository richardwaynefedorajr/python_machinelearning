import random
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
sns.set(rc={"figure.dpi":600, 'savefig.dpi':600})

# hyperparameters
w = torch.tensor([1.7]) #, -3.4])
b = 5.3
noise=0.5
num_samples = 100
epochs = 250
lr = 0.01

# generate data
X = torch.randn(num_samples, len(w))
noise = torch.randn(num_samples, 1) * noise
y = torch.matmul(X, w.reshape((-1, 1))) + b + noise

# create model
model = nn.LazyLinear(1)
model.weight.data.normal_(0, 0.01)
model.bias.data.fill_(0)

# loss function and optimizer
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = lr)

# fit
for epoch in range(epochs):
    y_hat = model(X)
    loss = loss_func(y_hat, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.item()))

# predict
y_hat = model(X)

# evaluate
squared_error = np.square(y - y_hat.detach()).numpy()

## add data to plot
fig = plt.figure()
ax = plt.axes()
df = pd.DataFrame(data=np.c_[X[:,0], y, y_hat.detach()], columns=['X','y','y_hat'])
sns.scatterplot(data=df, x='X', y='y', ax=ax, edgecolor='black', size=np.abs(y-y_hat.detach())[:,0], c='greenyellow')
# sns.scatterplot(data=df, x='X', y='y_hat', ax=ax, marker='o', edgecolor='black', s=2, c='darkgreen')
ax.plot(X[:,0], y_hat.detach(), color='green', linestyle='solid', linewidth=1, label='y_hat')
ax.set_title('Pytorch linear regression [MSE = '+str(round(squared_error.mean(),2))+']')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(fontsize=8, title_fontsize=8, title='marker size for\nerror values')
plt.savefig('linear_regression_torch.svg', bbox_inches='tight')
plt.show()