## Standard libraries
import pickle
import os
from copy import deepcopy
from tkinter import Image
from turtle import forward

## Imports for plotting
import matplotlib.pyplot as plt
plt.set_cmap('cividis')
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.set()
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

HIDDEN_DIM = 128;
BATCH_SIZE = 128;
LR = 5e-4;
TEMPERATURE = 0.07;
WEIGHT_DECAY = 1e-4;
MAX_EPOCHS = 500;
TRAIN_CONTRASTIVE = True;
DBG = False;
IMAGE_SIZE = 96

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "../data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "../saved_models/tutorial17"
# In this notebook, we use data loaders with heavier computational processing. It is recommended to use as many
# workers as possible in a data loader, which corresponds to the number of CPU cores
NUM_WORKERS = 2

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
print("Number of workers:", NUM_WORKERS)

import urllib.request
from urllib.error import HTTPError

def train(e, model, loader, optimizer, metrics = None):
    print(('\n' + '%10s'*3) %('Epoch', 'Loss', 'Acc'));
    pbar = enumerate(loader);
    pbar = tqdm(pbar, total=len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}');
    epochs_loss = [];
    epoch_acc = [];
    for idx, (batch) in pbar:
        imgs, lbl = batch
        imgs, lbl = imgs.to(device), lbl.to(device);
        
        output = model(imgs)

        
        loss = F.cross_entropy(output, lbl, reduction='mean');

        loss.backward();
        optimizer.step();
        model.zero_grad(set_to_none=True);

        f1 = metrics(output, lbl);
        epoch_acc.append(f1.item());
        epochs_loss.append(loss.item());
        

        pbar.set_description(("%10s" + "%10.4g"*2) %(e, np.mean(epochs_loss), np.mean(epoch_acc)))
        pass

    return np.mean(epochs_loss);

def valid(e, model, loader,):
    print(('\n' + '%10s'*3) %('Valid: Epoch', 'Loss', 'Acc'));
    pbar = enumerate(loader);
    pbar = tqdm(pbar, total=len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}');
    epochs_loss = [];
    epoch_acc = [];
    with torch.no_grad():
        for idx, (batch) in pbar:
            imgs, _ = batch
            imgs = torch.cat(imgs, dim=0)
            imgs = imgs.to(device);

            # Encode all images
            feats = model(imgs)
            # Calculate cosine similarity
            cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
            # Mask out cosine similarity to itself
            self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
            cos_sim.masked_fill_(self_mask, -9e15)
            # Find positive example -> batch_size//2 away from the original example
            pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
            # InfoNCE loss
            cos_sim = cos_sim / TEMPERATURE
            nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
            nll = nll.mean()

            loss = nll.item();

            # Get ranking position of positive example
            comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
                                cos_sim.masked_fill(pos_mask, -9e15)],
                                dim=-1)
            sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
            acc = (sim_argsort == 0).float().mean().item();

            epochs_loss.append(loss);
            epoch_acc.append(acc);

            pbar.set_description(("%10s" + "%10.4g"*2) %(e, np.mean(epochs_loss), np.mean(epoch_acc)))

    return np.mean(epochs_loss), np.mean(epoch_acc);



if __name__ == "__main__":

    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    train_data = STL10(root=DATASET_PATH, split='train', download=True,
                                transform=train_transforms)
    total = len(train_data);
    data3 = torch.utils.data.random_split(train_data, [total//10, total-total//10])[0]
    train_loader = data.DataLoader(data3, batch_size=BATCH_SIZE, shuffle=True,
                                    drop_last=False, pin_memory=True, num_workers=0)
    # val_loader = data.DataLoader(train_data_contrast, batch_size=BATCH_SIZE, shuffle=False,
    #                                 drop_last=False, pin_memory=True, num_workers=0)
    
    model = ResNet(3,10);
    model = model.to(device);
    optimizer = optim.Adam(model.parameters(), LR);
    f1_estimator = F1Score(task = 'multiclass').to(device);
    summary_writer = SummaryWriter('exp');
    best_loss = 1e10;
    best_model = None;
    for e in range(MAX_EPOCHS):
        model.train();
        train_loss = train(e, model, train_loader, optimizer, f1_estimator);
        #model.eval();
        #loss, acc = valid(e, model, val_loader);

        summary_writer.add_scalar('Train/Loss', train_loss, e);
        #summary_writer.add_scalar('Valid/Loss', loss, e);
        #summary_writer.add_scalar('Valid/Acc', acc, e);

        # if loss < best_loss:
        #     best_loss = loss;
        #     best_model = model.state_dict();
        #     pickle.dump(best_model, open('ckpt.pt', 'wb'))
        # lr_scheduler.step();