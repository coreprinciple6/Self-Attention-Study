## Reference: https://github.com/lukemelas/PyTorch-Pretrained-ViT
#### Get ViT-PyTorch using the below command
# !pip
# install - -upgrade
# pytorch - pretrained - vit

# !pip install --upgrade pytorch-pretrained-vit


print("running")

import json
from PIL import Image
import PIL
import torchvision
import torch
from torchvision import *
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

from pytorch_pretrained_vit import ViT

batch_size = 4
learning_rate = 1e-3
IMAGE_SIZE = 256
NUM_CLASSES = 2
lr = 0.001
n_epochs = 30
print_every = 1000
train_path = 'dataset/Train'
test_path = 'dataset/Val'

model_name = 'B_16_imagenet1k'
model = ViT(model_name, pretrained=True)

print(model)

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(model.image_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

train_dataset = datasets.ImageFolder(root=train_path, transform=transforms)
test_dataset = datasets.ImageFolder(root=test_path, transform=transforms)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

cnt = 5
print(random.randint(0, 100))

for name, child in model.named_children():
    print(name, child)
    cnt -= 1
    if cnt > 1:
        for name2, params in child.named_parameters():
            print(name2)
            params.requires_grad = False

model.last_linear = nn.Sequential(nn.Linear(in_features=128, out_features=2))

model = model.cuda() if device else model

# hyperparams
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.999),
                             eps=1e-08, weight_decay=0.1, amsgrad=False)
use_cuda = torch.cuda.is_available()

print(use_cuda)


def accuracy(out, labels):
    _, pred = torch.max(out, dim=1)
    return torch.sum(pred == labels).item()


# training

valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(train_dataloader)

torch.cuda.empty_cache()

for epoch in range(1, n_epochs + 1):
    running_loss = 0.0
    correct = 0
    total = 0
    print(f'Epoch {epoch}\n')
    for batch_idx, (data_, target_) in enumerate(train_dataloader):
        data_, target_ = data_.to(device), target_.to(device)
        optimizer.zero_grad()

        outputs = model(data_)
        loss = criterion(outputs, target_)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred == target_).item()
        total += target_.size(0)
        if (batch_idx) % print_every == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss / total_step)
    print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct / total):.4f}')
    batch_loss = 0
    total_t = 0
    correct_t = 0

    with torch.no_grad():
        model.eval()
        for data_t, target_t in (test_dataloader):
            data_t, target_t = data_t.to(device), target_t.to(device)
            outputs_t = model(data_t)
            loss_t = criterion(outputs_t, target_t)
            batch_loss += loss_t.item()
            _, pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(pred_t == target_t).item()
            total_t += target_t.size(0)
        val_acc.append(100 * correct_t / total_t)
        val_loss.append(batch_loss / len(test_dataloader))
        network_learned = batch_loss < valid_loss_min
        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')

        if network_learned:
            valid_loss_min = batch_loss
            torch.save(model.state_dict(), 'viT_model.pt')
            print('Improvement-Detected, save-model')
    model.train()

# plot performance
fig = plt.figure(figsize=(20, 10))
plt.title("Train-Validation Accuracy")
plt.plot(train_acc, label='train')
plt.plot(val_acc, label='validation')
plt.xlabel('num_epochs', fontsize=12)
plt.ylabel('accuracy', fontsize=12)
plt.legend(loc='best')
plt.show()
