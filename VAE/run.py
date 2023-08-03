import torch
import numpy as np
import torch.nn as nn
import torchvision.datasets as datasets
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import cv2
import os
from torchvision.utils import make_grid, save_image

LR = 1e-3;
EPOCHS = 3000;
BATCH_SIZE = 32;
IMAGE_DIMENSION = 784;
NEURAL_NETWORK_DIMENSION = 512;
DEVICE = 'cuda';
LATENT_VARIABLE_DIMENSION = 2;

def loss_func(original_image, reconstructed_image, std_layer, mean_layer):
    fidelity_loss = original_image * torch.log(1e-10+reconstructed_image) + (1-original_image)*torch.log(1e-10+1-reconstructed_image);
    fidelity_loss = -torch.sum(fidelity_loss, 1)
    kl_div_loss = 1+std_layer - torch.square(mean_layer) - torch.exp(std_layer);
    kl_div_loss = -0.5*torch.sum(kl_div_loss, 1);

    loss = kl_div_loss + fidelity_loss;
    return torch.mean(loss);

     

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Sequential(
        nn.Linear(IMAGE_DIMENSION, NEURAL_NETWORK_DIMENSION),
        nn.ReLU(),
        nn.Linear(NEURAL_NETWORK_DIMENSION, NEURAL_NETWORK_DIMENSION),
        )
        self.mean_layers = nn.Linear(NEURAL_NETWORK_DIMENSION, LATENT_VARIABLE_DIMENSION);
        self.std_layers = nn.Linear(NEURAL_NETWORK_DIMENSION, LATENT_VARIABLE_DIMENSION);

        self.dec = nn.Sequential(
            nn.Linear(LATENT_VARIABLE_DIMENSION, NEURAL_NETWORK_DIMENSION),
            nn.ReLU(),
            nn.Linear(NEURAL_NETWORK_DIMENSION, IMAGE_DIMENSION),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        enc = self.enc(x);
        eps = torch.randn((x.shape[0], LATENT_VARIABLE_DIMENSION)).to(DEVICE);
        std_layer = self.std_layers(enc);
        mean_layer = self.mean_layers(enc);
        latent_layer = mean_layer + eps * torch.exp(std_layer*0.5);
        dec = self.dec(latent_layer);
        return std_layer, mean_layer, dec;

def test():
    net = Net();
    out = net(torch.randn(3,IMAGE_DIMENSION));
    print(out[2].shape);


if __name__ == "__main__":
    net = Net();
    dataset = datasets.MNIST(root='dataset/', train=True, transform = transforms.Compose([transforms.ToTensor()]), download = True);
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True);
    optimizer = optim.Adam(net.parameters(), LR);
    net.to(DEVICE);

    if os.path.exists('samples') is False:
        os.mkdir('samples');
    
    sample_save_interval = 1;

    for epoch in range(EPOCHS):
        print(('\n' + '%10s'*2) %('Epoch', 'Loss',));
        pbar = tqdm(enumerate(train_loader), total= len(train_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        epoch_loss = [];
        for i, (x,_) in pbar:
            x = x.to(DEVICE);
            x = x.view(-1, IMAGE_DIMENSION);
            std_layer, mean_layer, dec = net(x);
            loss = loss_func(x, dec, std_layer, mean_layer);
            loss.backward();
            optimizer.step();
            net.zero_grad(set_to_none=True);
            epoch_loss.append(loss.item());
            pbar.set_description(('%10s' + '%10.4g'*1) % (epoch, np.mean(epoch_loss)))
        
        if epoch % sample_save_interval == 0:
            net.eval();
            batch,_ = next(iter(train_loader));
            batch = batch.to(DEVICE);
            batch = batch.view(-1, IMAGE_DIMENSION);
            _, _, rec = net(batch);

            rec = rec.view(-1, 1, 28, 28);
            grid = make_grid(rec, nrow=rec.shape[0]);
            save_image(grid, f'samples/{epoch}.png');
            
            net.train();