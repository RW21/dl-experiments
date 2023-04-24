import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        # (1, 28, 28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        # (32, 26, 26)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (32, 13, 13)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        # (64, 11, 11)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # (64, 5, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)

        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)

        x = self.fc1(x.view(x.shape[0], -1))
        # Don't include softmax here
        return x


def train(train_dataloader, model, loss_fn, optimizer, device):

    for (data, labels) in train_dataloader:
        data, labels = data.to(device), labels.to(device)
        pred = model(data)

        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss

def accuracy(test_dataloader, model, device):
    correct, total = 0, 0

    with torch.no_grad(): 
        for (data, labels) in test_dataloader:
            data, labels = data.to(device), labels.to(device)
            pred = model(data)
            pred = torch.argmax(pred, dim=1)
            total += pred.shape[0]
            correct += (pred == labels).sum()

    accuracy = (correct / total).item() * 100
    return accuracy
