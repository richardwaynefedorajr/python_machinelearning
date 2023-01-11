import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.nn import functional as F

def initializeWeights(layer):
    if isinstance(layer, nn.LazyConv2d) or isinstance(layer, nn.LazyLinear):
        torch.nn.init.xavier_normal_(layer.weight)
        # torch.nn.init.normal_(layer.weight)
        if layer.bias is not None:
                layer.bias.data.zero_()

# hyperparameters
batch_size = 256
output_classes = 10
lr = 0.25 # 84.765625% (no batch norm)
epochs = 10

# activation and loss
activation_function = nn.ReLU()
# loss_function = nn.NLLLoss()
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
                    # nn.LogSoftmax())

# initialize
model.apply(initializeWeights)
# optimizer = optim.Adam(model.parameters(), lr=lr)
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
        wrong = y_hat.argmax(axis=1) != y
        y, y_hat = y[wrong], y_hat[wrong]
        accuracy = 1 - len(y_hat)/batch_size
    print('epoch {}, loss {}, accuracy {}'.format(epoch, loss.item(), accuracy))
    

# predict
X, y = next(iter(torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=0)))
y_hat = model(X).argmax(axis=1)

# evaluate
wrong = y_hat != y
X, y, y_hat = X[wrong], y[wrong], y_hat[wrong]
accuracy = 1 - len(y_hat)/batch_size
print(accuracy)