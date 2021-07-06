import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import VGG_net

# hyperparameters
in_channel = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

# load data 
train_dataset = datasets.ImageFolder(root='../dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.ImageFolder(root='../dataset/', train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

device = ('cuda' if torch.cuda.is_available() else 'cpu')

# initialize network
model = VGG_net().to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # get data from cuda
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        # get data to correct shape
        #data = data.reshape(data.shape[0], -1)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        print(loss)
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
            
            #x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            
        print('got {} / {} with accuracy: {}'.format(num_correct, num_samples, float(num_correct)/num_samples))
        model.train()
    
checkAccuracy(train_loader, model)
checkAccuracy(test_loader, model)

