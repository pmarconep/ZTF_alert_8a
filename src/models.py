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
        return z
        
    def forward(self, x):
        z = self.only_encoder(x)
        reconstruction = self.only_decoder(z)
        return reconstruction
    
def ae_loss_function(recon_x, x):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    return recon_loss



class RNN(nn.Module):
    def __init__(self, m_type, n_stamps, latent_dim, hidden_dim, num_layers, num_class, name):
        super(RNN, self).__init__()
        
        self.name = name # for plotting purposes
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_stamps = n_stamps

        if m_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        elif m_type == 'GRU':
            self.rnn = nn.GRU(input_size=latent_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        elif m_type == 'RNN':
            self.rnn = nn.RNN(input_size=latent_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        else:
            raise ValueError("Invalid RNN type.")
            
        self.fc1 = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(0.5) # Valor "arbitrario"
        self.fc3 = nn.Linear(latent_dim, num_class)


    def forward(self, x):
        x, _ = self.rnn(x)
        x = x[:,-1,:]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def rnn_loss_function(predictions, labels, weights=None):
    if weights != None:
        criterion = nn.CrossEntropyLoss(weight=weights)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(predictions, labels)
    return loss


class LinearClassifier(nn.Module):
    def __init__(self, latent_dim, num_classes, name):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(latent_dim, num_classes)
        self.name = name
        
    def forward(self, z):
        out = self.linear(z)
        return out

from torch.utils.data import TensorDataset
import numpy as np

def get_latent_features(models, train_dataset, val_dataset, test_dataset):

    model = models[0]
    
    train = np.empty(len(models), dtype=object)
    val = np.empty(len(models), dtype=object)
    test = np.empty(len(models), dtype=object)

    train = TensorDataset(model.encoder(train_dataset.tensors[0]).detach(), train_dataset.tensors[1].detach())
    val = TensorDataset(model.encoder(val_dataset.tensors[0]).detach(), val_dataset.tensors[1].detach())
    test = TensorDataset(model.encoder(test_dataset.tensors[0]).detach(), test_dataset.tensors[1].detach())

    return train, val, test