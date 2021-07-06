import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import EfficientNet


transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
)

# hyperparameters
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# hyperparameters
in_channel = 3
num_classes = len(classes)
learning_rate = 0.001
batch_size = 64
num_epochs = 5
version = 'b0'

# load data
train_dataset = torchvision.datasets.CIFAR10(root='../dataset', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='../dataset', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = EfficientNet(version=version, num_classes=num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        
        # backward
        optimizer.zero_grad()
        loss.backward()

        # statistics
        running_loss += loss.item()
        if batch_idx % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

        # gradient descent
        optimizer.step()

PATH = 'models/cifar_net.pth'
torch.save(model.state_dict(), PATH)

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