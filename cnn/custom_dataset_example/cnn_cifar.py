import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
import sys
from torchvision.transforms.transforms import Normalize
from torchvision.utils import save_image
from CustomDataset import CatsAndDogsDataset

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
in_channel = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 64
num_epochs = 15

# load data
transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.RandomCrop((224,224)),
    transforms.RandomRotation(degrees=45),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[00.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
])

dataset = CatsAndDogsDataset(csv_file='cats_dogs.csv',
                             root_dir='cats_dogs_resized',
                             transform=transforms)

img_num = 0
for img, label in dataset:
    save_image(img, 'cats_dogs_resized/img'+str(img_num) + '.png')
    img_num += 1


train_set, test_set = torch.utils.data.random_split(dataset, [7, 3])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# load prerain model and modify it
model = torchvision.models.googlenet(pretrained=True)
model.to(device)

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
    '''
    if loader.dataset.train:
        print('checking accuracy on training data')
    else:
        print('checking accuracy on test data')
    '''

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

