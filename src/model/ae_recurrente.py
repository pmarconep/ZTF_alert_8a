import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
    
class AE_recurrent(nn.Module):
    def __init__(self, latent_dim, n_channels, sequence_length, name, hidden_dim, num_layers):
        super(AE_recurrent, self).__init__()
        
        self.name = name # for plotting purposes
        self.latent_dim = latent_dim
        self.n_channels = n_channels
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder
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
        
        # RNN Layer
        self.rnn = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64*6*6),
            nn.ReLU(),
            nn.Unflatten(1, (64, 6, 6)),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(size=(11, 11), mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(size=(21, 21), mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, n_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def only_encoder(self, x):
        batch_size, sequence_length, channels, height, width = x.size()
        x = x.view(batch_size * sequence_length, channels, height, width)  # flatten sequences
        print("x",x.size())
        z = self.encoder(x)
        z = z.view(batch_size, sequence_length, -1)  # reshape to (batch, seq_len, latent_dim)
        print("z",z.size())
        return z
    
    def only_decoder(self, z):
        # Decodifica a partir del último estado oculto de la RNN
        return self.decoder(z)

    def forward(self, x):
        z = self.only_encoder(x)
        _, (h_n, _) = self.rnn(z)  # h_n contiene el último estado oculto de la LSTM
        h_n = h_n[-1]  # tomar el último estado de la última capa (dimensión de hidden_dim)
        reconstruction = self.only_decoder(h_n)
        
        return reconstruction
    
def loss_function(recon_x, x):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    return recon_loss
