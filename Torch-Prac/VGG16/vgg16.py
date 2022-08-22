import enum
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

BATCH_SIZE = 1024
NUM_CLASSES = 10
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5
IN_CHANNEL = 3


class Identity(nn.Module):
    def __init__(self) -> None:
        super(Identity, self).__init__()

    def forward(self, x):
        return x


model = torchvision.models.vgg16(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
# Repalcing avgpool layer
model.avgpool = Identity()

# Replacing classifier layers
model.classifier = nn.Sequential(
    nn.Linear(512, 100), nn.ReLU(), nn.Linear(100, 10))
print(model)

train_dataset = datasets.CIFAR10(
    root='data', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
test_dataset = datasets.CIFAR10(
    root='data', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train(model, loader, criterion, optimizer):
    for epoch in range(NUM_EPOCHS):
        losses = []

        for batch_idx, (data, targets) in enumerate(loader):
            scores = model(data)
            loss = criterion(scores, targets)

            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if batch_idx % 100 == 0:
            loss, current = loss.item(), batch_idx * len(data)
            print(f'loss: {loss:>7f}')
