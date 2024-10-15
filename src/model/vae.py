import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
    
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        self.fc_mu = nn.Linear(64*6*6, latent_dim)
        self.fc_logvar = nn.Linear(64*6*6, latent_dim)
        
        self.decoder = nn.Sequential(
            
            # FCL
            nn.Linear(latent_dim, 64*6*6),
            nn.ReLU(),
            
            # Reshape to (64, 6, 6)
            nn.Unflatten(1, (64, 6, 6)),
            
            #1st Convolutional Layer
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            #2nd Convolutional Layer
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Nearest Neighbour interpolation
            nn.Upsample(size=(11,11), mode='nearest'),
            
            #3rd Convolutional Layer
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            #4th Convolutional Layer
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Nearest Neighbour interpolation
            nn.Upsample(size=(21,21), mode='nearest'),
            
            #5th Convolutional Layer
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  
            
            #6th Convolutional Layer
            nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1), # Tengo dudas con esta última capa
            nn.ReLU()            
        )
        
        self.fc_sigma = nn.Sequential(
                nn.Linear(latent_dim, 36),
                nn.ReLU(),
                nn.Linear(36, 1),
                nn.Sigmoid()  # Constrain variance to be between small positive values
            )
        
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparametrize(mu, logvar)
        sigma = self.fc_sigma(z)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar, sigma
    
def loss_function(recon_x, x, mu, logvar, sigma):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / (2 * sigma) + torch.log(sigma)
    # Kullback-Leibler divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss