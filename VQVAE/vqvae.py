import numpy as np
import torch
import os
import torch.nn as nn
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm

class VQVAE(nn.Module):
    """Vector Quantized Variational Autoencoder (VQ-VAE) model."""
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 4, 4, 2, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )

        self.pre_quant_conv = nn.Conv2d(4, 2, kernel_size=1)
        self.embedding = nn.Embedding(64, 2)
        self.post_quant_conv = nn.Conv2d(2, 4, 1)

        self.beta = 0.2

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        encoder_output = self.encoder(x)
        quant_input = self.pre_quant_conv(encoder_output)
        B, C, H, W = quant_input.shape
        quant_input = quant_input.permute(0, 2, 3, 1).reshape(B, -1, C)

        dist = torch.cdist(quant_input, self.embedding.weight[None, :].repeat(B, 1, 1))
        min_encoding_indices = torch.argmin(dist, dim=-1)
        quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indices.view(-1))

        commitment_loss = torch.mean((quant_out.detach() - quant_input) ** 2)
        codebook_loss = torch.mean((quant_out - quant_input.detach()) ** 2)
        quant_out = quant_input + (quant_out - quant_input).detach()

        quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)
        decoder_input = self.post_quant_conv(quant_out)
        output = self.decoder(decoder_input)
        recon_loss = torch.mean((x - output) ** 2)
        quantize_loss = codebook_loss + self.beta * commitment_loss + recon_loss
        return output, quantize_loss

    def generate_sample(self):
        """Generate a sample from the learned embedding space."""
        sample = torch.randn((1, 49, 2)).to('cuda')
        dist = torch.cdist(sample, self.embedding.weight[None, :].repeat(sample.size(0), 1, 1))
        min_encoding_indices = torch.argmin(dist, dim=-1)
        quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indices.view(-1))
        quant_out = quant_out.reshape((1, 2, 7, 7))
        decoder_input = self.post_quant_conv(quant_out)
        return self.decoder(decoder_input)

def save_samples(model):
    """Save generated samples to disk."""
    with torch.no_grad():
        out = model.generate_sample()
        save_image(out.squeeze(), 'samples/test.png')

if __name__ == "__main__":
    model = VQVAE().to('cuda')
    train_data = MNIST(root='data', train=True, transform=ToTensor(), download=True)
    train_loader = DataLoader(train_data, 128, True, num_workers=0, pin_memory=True)
    optimizer = optim.AdamW(model.parameters(), 3e-4)

    os.makedirs('samples', exist_ok=True)
    sample_freq = 10
    epochs = 500
    
    for e in range(epochs):
        epoch_loss = []
        for img, _ in tqdm(train_loader):
            img = img.to('cuda')
            out, loss = model(img)
            loss.backward()
            optimizer.step()
            model.zero_grad(set_to_none=True)
            epoch_loss.append(loss.item())
            if (e + 1) % sample_freq == 0:
                save_samples(model)
        print(f'Epoch: {e+1}, Loss: {np.mean(epoch_loss)}')
