import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
    
class AE(nn.Module):
    def __init__(self, latent_dim, n_channels, name):
        super(AE, self).__init__()
        
        self.name = name #for plotting purposes

        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1),
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
            nn.Linear(64*6*6, latent_dim)
        )
        
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
            nn.Conv2d(64, n_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def only_encoder(self, x):
        return self.encoder(x)
    
    def only_decoder(self, z):
        return self.decoder(z)
    
    def time_sequence(self, x):
        batch_size, sequence_length, channels, height, width = x.size()
        x = x.view(batch_size * sequence_length, channels, height, width)
        z = self.encoder(x)
        z = z.view(batch_size, sequence_length, -1)
        z = z.view(batch_size, -1)
        return z
        
    def forward(self, x):
        z = self.only_encoder(x)
        reconstruction = self.only_decoder(z)
        return reconstruction
    
def loss_function(recon_x, x):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    return recon_loss