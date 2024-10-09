from pathlib import Path
import time

import torch
from torch import nn
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class VAE(nn.Module):
    def __init__(self, latent_dim=21, img_size=21):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size

        # Encoder network
        self.encoder = nn.Sequential(

            # 1st Convolutional Layer
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # Output: (64, 21, 21)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            

            # 2nd Convolutional Layer
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # Output: (64, 11, 11)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 3rd Convolutional Layer
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # Output: (64, 11, 11)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 4th Convolutional Layer
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # Output: (64, 6, 6)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Latent space parameters
        self.flatter = nn.Flatten() # Flatten the output of the encoder 6*6*64
        self.fc_mu = nn.Linear(in_features=64 * 6 * 6, out_features=latent_dim) # 
        self.fc_logvar = nn.Linear(in_features=64 * 6 * 6, out_features=latent_dim)

        # Decoder network
        self.fc_decoder = nn.Linear(latent_dim, 64 * 6 * 6)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        )
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),  # Output: (64, 6, 6)
        #     nn.ReLU(),
        #     nn.BatchNorm2d(64),
        #     nn.Upsample(scale_factor=2),  # Output: (64, 12, 12)
        #     nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),  # Output: (64, 12, 12)
        #     nn.ReLU(),
        #     nn.BatchNorm2d(64),
        #     nn.Upsample(scale_factor=2),  # Output: (64, 24, 24)
        #     nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1),  # Output: (1, 21, 21)
        #     nn.Sigmoid()  # Output should match original image size
        # )

        # Auto-regularization network (for estimating variance)
        self.fc_sigma = nn.Sequential(
            nn.Linear(latent_dim, 36),
            nn.ReLU(),
            nn.Linear(36, 1),
            nn.Sigmoid()  # Constrain variance to be between small positive values
        )
        
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decoder(z)
        h = h.view(-1, 64, 6, 6)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    # def encode(self, x):
    #     h = self.encoder(x)
    #     h = h.view(-1, 64 * 6 * 6)  # Flatten
    #     mu = self.fc_mu(h)
    #     logvar = self.fc_logvar(h)
    #     return mu, logvar

    # def reparametrize(self, mu, logvar):
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return mu + eps * std

    # def decode(self, z):
    #     h = self.fc_decoder(z)
    #     h = h.view(-1, 64, 6, 6)
    #     x_recon = self.decoder(h)
    #     return x_recon

    # def forward(self, x):
    #     mu, logvar = self.encode(x)
    #     z = self.reparametrize(mu, logvar)
    #     sigma = self.fc_sigma(z)  # Estimate variance as part of auto-regularization
    #     x_recon = self.decode(z)
    #     return x_recon, mu, logvar, sigma

def vae_loss_function(recon_x, x, mu, logvar, sigma):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / (2 * sigma) + torch.log(sigma)
    # Kullback-Leibler divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss