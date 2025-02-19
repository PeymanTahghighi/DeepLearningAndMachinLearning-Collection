## Standard libraries
import pickle
import os
from copy import deepcopy

## Imports for plotting
import matplotlib.pyplot as plt
plt.set_cmap('cividis')
import matplotlib
import seaborn as sns
import numpy as np

## tqdm for loading bars
from tqdm import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import cv2

## Torchvision
import torchvision
from torchvision.datasets import STL10
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from resnet import ResNet
from torchmetrics import F1Score

HIDDEN_DIM = 128
BATCH_SIZE = 128
LR = 5e-4
TEMPERATURE = 0.07
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 500
IMAGE_SIZE = 96

DATASET_PATH = "../data"
CHECKPOINT_PATH = "../saved_models/tutorial17"
NUM_WORKERS = 2

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
print("Number of workers:", NUM_WORKERS)

def train(e, model, loader, optimizer, metrics=None):
    """Train function for one epoch."""
    print(('%10s' * 3) % ('Epoch', 'Loss', 'Acc'))
    pbar = tqdm(enumerate(loader), total=len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    epochs_loss = []
    epoch_acc = []
    for idx, (batch) in pbar:
        imgs, lbl = batch
        imgs, lbl = imgs.to(device), lbl.to(device)
        output = model(imgs)
        loss = F.cross_entropy(output, lbl, reduction='mean')
        loss.backward()
        optimizer.step()
        model.zero_grad(set_to_none=True)
        f1 = metrics(output, lbl)
        epoch_acc.append(f1.item())
        epochs_loss.append(loss.item())
        pbar.set_description(("%10s" + "%10.4g" * 2) % (e, np.mean(epochs_loss), np.mean(epoch_acc)))
    return np.mean(epochs_loss)

if __name__ == "__main__":
    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    train_data = STL10(root=DATASET_PATH, split='train', download=True, transform=train_transforms)
    total = len(train_data)
    data3 = torch.utils.data.random_split(train_data, [total // 10, total - total // 10])[0]
    train_loader = data.DataLoader(data3, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
    
    model = ResNet(3, 10).to(device)
    optimizer = optim.Adam(model.parameters(), LR)
    f1_estimator = F1Score(task='multiclass').to(device)
    summary_writer = SummaryWriter('exp')
    best_loss = float('inf')
    best_model = None
    
    for e in range(MAX_EPOCHS):
        model.train()
        train_loss = train(e, model, train_loader, optimizer, f1_estimator)
        summary_writer.add_scalar('Train/Loss', train_loss, e)
