import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
    
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

def loss_function(predictions, labels, weights=None):
    if weights != None:
        criterion = nn.CrossEntropyLoss(weight=weights)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(predictions, labels)
    return loss