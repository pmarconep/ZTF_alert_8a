import torch.nn as nn

class FinalModel(nn.Module):
    def __init__(self, latent_dim, n_channels, rnn_type, hidden_dim, num_layers, dropout_prob, n_classes, name = 'None', description = 'None'):
        super(FinalModel, self).__init__()

        self.name = name
        self.description = description

        assert type(latent_dim) == int, "Latent dimension must be an int"
        assert latent_dim > 0, "Latent dimension must be grater than 0"
        
        assert type(n_channels) == int, "Number of channels must be an int"
        assert n_channels > 0, "Number of channels must be grater than 0"

        self.n_channels = n_channels
        self.latent_dim = latent_dim

        #encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([64, 21, 21]),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([64, 11, 11]),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([64, 11, 11]),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([64, 6, 6]),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Flatten(),
            nn.Linear(64*6*6, latent_dim),
            nn.Dropout(dropout_prob)
        )

        #decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64*6*6),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Unflatten(1, (64, 6, 6)),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([64, 6, 6]),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([64, 6, 6]),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Upsample(size=(11,11), mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([64, 11, 11]),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([64, 11, 11]),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Upsample(size=(21,21), mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([64, 21, 21]),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Conv2d(64, n_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # RNN
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=latent_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        elif rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size=latent_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        else:
            raise ValueError("Invalid RNN type.")
        self.dropout_rnn = nn.Dropout(0.25)
        self.fc3 = nn.Linear(latent_dim, n_classes)

    #autoencoder
    def only_encoder(self, x):
        return self.encoder(x)
    
    def only_decoder(self, z):
        return self.decoder(z)
    
    def reconstruction(self, x):
        z = self.only_encoder(x.view(-1, self.n_channels, 21, 21))
        reconstruction = self.only_decoder(z)
        return reconstruction
    

    #rnn
    def rnn_encode(self, x):
        size = x.shape[0]
        x = self.only_encoder(x.view(-1, self.n_channels, 21, 21))
        x, _ = self.rnn(x.view(size, 5, self.latent_dim))
        x = self.dropout_rnn(x)
        reconstruction = self.only_decoder(x.reshape(-1, self.latent_dim))
        return reconstruction

    def rnn_latent(self, x):
        size = x.shape[0]
        x = self.only_encoder(x.view(-1, self.n_channels, 21, 21))
        x, _ = self.rnn(x.view(size, 5, self.latent_dim))
        x = self.dropout_rnn(x)
        return x[:, -1, :]

    def rnn_classifier(self, x):
        size = x.shape[0]
        x = self.only_encoder(x.view(-1, self.n_channels, 21, 21))
        x, _ = self.rnn(x.view(size, 5, self.latent_dim))
        x = self.dropout_rnn(x)
        x = self.fc3(x[:, -1, :])
        return x
    
    def forward(self, x):
        size = x.shape[0]
        x = self.only_encoder(x.view(-1, self.n_channels, 21, 21))
        x, _ = self.rnn(x.view(size, 5, self.latent_dim))
        x = self.dropout_rnn(x)
        x = self.fc3(x[:, -1, :])
        return x
    
def ae_loss_function(recon_x, x):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    return recon_loss

def rnn_loss_function(predictions, labels):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(predictions, labels)
    return loss

def combined_loss_function(recon_x, x, predictions, labels, alpha=0.5):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    class_loss = nn.CrossEntropyLoss()(predictions, labels)
    return alpha * recon_loss + (1 - alpha) * class_loss