import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# Define model
class NN(nn.Module):
    def __init__(self, input_size, num_classes):  # 28 x 28
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparams
input_size = 784
num_classes = 10
learning_rate = .001
batch_size = 64
num_epochs = 10

# Load Data
training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    # download=True,
    transform=transforms.ToTensor(),
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    # download=True,
    transform=transforms.ToTensor(),
)


train_dataloader = DataLoader(
    training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


for X, y in test_dataloader:
    print(f'Shape of X [N, C, H, W]: {X.shape}')
    print(f'Shape of y: {y.shape}, {y.dtype}')
    break


# initialize model
model = NN(input_size, num_classes).to(device)
print(model)


# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# Train Network

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):

        X = X.reshape(X.shape[0], -1)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.reshape(X.shape[0], -1)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f'Test Error : {(100*correct):>0.1f} \n Avg loss: {test_loss:>8f} \n')


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n {20*'-'}")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Done!")

# SAVING MODELS
# torch.save(model.state_dict(), "model.pth")

# LOADING MODELS
# model = NeuralNetwork()
# model.load_state_dict(torch.load("model.pth"))


# Alternate training method
# for epoch in range(num_epochs):
#     for batch_idx, (data, targets), in enumerate(train_dataloader):

#         # Get data to device
#         data = data.to(device)
#         targets = targets.to(device)

#         # Get to correct shape
#         data = data.reshape(data.shape[0], -1)

#         scores = model(data)
#         loss = loss_fn(scores, targets)

#         # backward
#         optimizer.zero_grad()
#         loss.backward()

#         # gradient descent/Adam step
#         optimizer.step()

# Check accuracy on training and testing


# def check_accuracy(loader, model):
#     if loader.dataset.train:
#         print('Checking accuracy')
#     else:
#         print('Checking accuracy on test data')

#     num_correct = 0
#     num_samples = 0
#     model.eval()

#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device)
#             y = y.to(device)
#             x = x.reshape(x.shape[0], -1)

#             scores = model(x)
#             _, predictions = scores.max(1)
#             num_correct += (predictions == y).sum()
#             num_samples += predictions.size(0)

#         print(
#             f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

#     model.train()


# check_accuracy(train_dataloader, model)
# check_accuracy(test_dataloader, model)
