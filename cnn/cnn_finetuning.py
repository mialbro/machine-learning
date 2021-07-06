import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
in_channel = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 64
num_epochs = 5

import sys

class Identity(nn.Modeul):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x 


# load prerain model and modify it
model = torchvision.models.vgg16(pretrained=False)

for param in model.parameters():
    param.requires_grad = False

model.avgpool = Identity()
model.classifier = nn.Sequential(nn.Linear(512, 10),
                                 nn.ReLu(),
                                 nn.Linear(100, 10))
model.to(device)

# load data 
train_dataset = datasets.CIFAR10(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.CIFAR10(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # get data from cuda
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad() # set all gradients to zero for each batch so previous batch results are not present
        loss.backward()

        # gradient descent
        optimizer.step()


# check accuracy on training and test
def checkAccuracy(loader, model):
    if loader.dataset.train:
        print('checking accuracy on training data')
    else:
        print('checking accuracy on test data')

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad(): # gradients are not computed
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            
        print('got {} / {} with accuracy: {}'.format(num_correct, num_samples, float(num_correct)/num_samples))
        model.train()
    
checkAccuracy(train_loader, model)
checkAccuracy(test_loader, model)

