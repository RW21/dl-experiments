import torch
from model import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
lr = 0.001
# Num of clients
K = 5
num_rounds = 3
# Fraction of clients to participate in each round
C = .5

class Client:
    def __init__(self, model, train_dataloader):
        self.train_dataloader = train_dataloader
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_log = []


fmnist_train = datasets.FashionMNIST(root='fmnist', train=True, download=True, transform=transforms.ToTensor())
fmnist_test = datasets.FashionMNIST(root='fmnist', train=False, download=True, transform=transforms.ToTensor())

splits = [1/K for _ in range(K)]
fmnist_train_split = torch.utils.data.random_split(fmnist_train, splits)

clients = []
# Create clients
for i in range(K):
    model = Net2().to(device)
    train_dataloader = torch.utils.data.DataLoader(fmnist_train_split[i], batch_size=batch_size, shuffle=True)

    c = Client(model, train_dataloader)
    clients.append(c)

global_model = Net2().to(device)

# Use global model weights as initial weights
for c in clients:
    c.model.load_state_dict(global_model.state_dict())

for round in range(num_rounds):
    print(f"Communication round {round + 1}")

    selected_clients = np.random.choice(clients, int(C * K), replace=False)

    for c in tqdm(selected_clients):
        c: Client
        # Only one epoch
        loss = train(c.train_dataloader, model=c.model, loss_fn=c.loss_fn, optimizer=c.optimizer, device=device)
        c.loss_log.append(loss)

    # Aggregate
    weight_sum = sum(len(c.train_dataloader) for c in selected_clients)
    for k, v in global_model.state_dict().items():
        # Reset global
        v *= 0
        for c in selected_clients:
            v += c.model.state_dict()[k] * len(c.train_dataloader) / weight_sum
            global_model.state_dict()[k] += v


test_dataloader = torch.utils.data.DataLoader(fmnist_test, batch_size=batch_size, shuffle=True)
accuracy = accuracy(test_dataloader=test_dataloader, model=global_model, device=device)
print(f"{accuracy=}")
