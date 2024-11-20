import torch.nn as nn

class FinalModel(nn.module):
    def __init__(self, latent_dim, n_channels, rnn_type, hidden_dim, num_layers, n_classes, name = 'None', description = 'None'):
        super(FinalModel, self).__init__()

        self.name = name
        self.description = description

        assert type(latent_dim) == int, "Latent dimension must be an int"
        assert latent_dim > 0, "Latent dimension must be grater than 0"
        
        assert type(n_channels) == int, "Number of channels must be an int"
        assert n_channels > 0, "Number of channels must be grater than 0"

        #encoder
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

        #decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64*6*6),
            nn.ReLU(),
            nn.Unflatten(1, (64, 6, 6)),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(size=(11,11), mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(size=(21,21), mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  
            nn.Conv2d(64, n_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=latent_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        elif rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size=latent_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        else:
            raise ValueError("Invalid RNN type.")
            
        self.fc1 = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(0.5) # Valor "arbitrario"
        self.fc3 = nn.Linear(latent_dim, n_classes)


    #autoencoder
    def only_encoder(self, x):
        return self.encoder(x)
    
    def only_decoder(self, z):
        return self.decoder(z)
    
    def reconstruction(self, x):
        z = self.only_encoder(x)
        reconstruction = self.only_decoder(z)
        return reconstruction
    

    #rnn
    def classifier(self, x):
        x, _ = self.rnn(x)
        x = x[:,-1,:]
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    

    #completo
    def forward(self, x):
        z = self.only_encoder(x)
        z, _ = self.rnn(z)
        z = z[:,-1,:]
        z = nn.functional.relu(self.fc1(z))
        z = self.dropout(z)
        z = self.fc3(z)
        return z
    
def ae_loss_function(recon_x, x):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    return recon_loss

def rnn_loss_function(predictions, labels, weights=None):
    if weights != None:
        criterion = nn.CrossEntropyLoss(weight=weights)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(predictions, labels)
    return loss