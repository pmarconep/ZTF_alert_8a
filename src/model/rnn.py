import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
    
class RNN(nn.Module):
    def __init__(self, n_stamps, latent_dim, hidden_dim, num_layers, num_class, name):
        super(RNN, self).__init__()
        
        self.name = name # for plotting purposes
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_stamps = n_stamps

        self.rnn = nn.LSTM(input_size=latent_dim*n_stamps, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, latent_dim*n_stamps)
        self.dropout = nn.Dropout(0.5) # Valor "arbitrario"
        self.fc2 = nn.Linear(latent_dim*n_stamps, latent_dim*n_stamps)
        self.fc3 = nn.Linear(latent_dim*n_stamps, num_class)


    def forward(self, x):
        x = x.view(-1, self.n_stamps, self.latent_dim)
        x, _ = self.rnn(x)
        x = x[:,-1,:]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        output = F.softmax(x, dim=1)
        return output


    
def loss_function(predictions, labels):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(predictions, labels)
    return loss

