
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

class EarlyStopping:
    def __init__(self, n_epochs_tolerance):
        self.n_epochs_tolerance = n_epochs_tolerance
        self.epochs_with_no_improvement = 0
        self.best_loss = np.inf

    def __call__(self, val_loss):
        # En cada llamada aumentamos el número de épocas en que no hemos mejorado
        self.epochs_with_no_improvement += 1

        if val_loss <= self.best_loss:
            # Si efectivamente mejoramos (menor loss de validación) reiniciamos el número de épocas sin mejora
            self.best_loss = val_loss
            self.epochs_with_no_improvement = 0

        # Retornamos True si debemos detenernos y False si aún no
        # Nos detenemos cuando el número de épocas sin mejora es mayor o igual que el número de épocas de tolerancia
        return self.epochs_with_no_improvement >= self.n_epochs_tolerance
    
class VAE(nn.Module):
    def __init__(self, latent_dim=21, img_size=21):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size

        # Encoder network
        self.encoder = nn.Sequential(

            # 1st Convolutional Layer
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # Output: (64, 21, 21)
            nn.ReLU(), # Activation function
            nn.BatchNorm2d(64), # Batch Normalization 

            # 2nd Convolutional Layer
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # Output: (64, 11, 11)
            nn.ReLU(),
            nn.BatchNorm2d(64),

            # 3rd Convolutional Layer
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # Output: (64, 11, 11)
            nn.ReLU(),
            nn.BatchNorm2d(64),

            # 4th Convolutional Layer
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # Output: (64, 6, 6)
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        # Latent space parameters
        # self.flatter = nn.Flatten()
        self.fc_mu = nn.Linear(64 * 6 * 6, latent_dim) # 
        self.fc_logvar = nn.Linear(64 * 6 * 6, latent_dim)

        # Decoder network
        self.fc_decoder = nn.Linear(latent_dim, 64 * 6 * 6)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),  # Output: (64, 6, 6)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2),  # Output: (64, 12, 12)
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),  # Output: (64, 12, 12)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2),  # Output: (64, 24, 24)
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1),  # Output: (1, 21, 21)
            nn.Sigmoid()  # Output should match original image size
        )

        # Auto-regularization network (for estimating variance)
        self.fc_sigma = nn.Sequential(
            nn.Linear(latent_dim, 36),
            nn.ReLU(),
            nn.Linear(36, 1),
            nn.Sigmoid()  # Constrain variance to be between small positive values
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(-1, 64 * 6 * 6)  # Flatten
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
        sigma = self.fc_sigma(z)  # Estimate variance as part of auto-regularization
        x_recon = self.decode(z)
        return x_recon, mu, logvar, sigma

def vae_loss_function(recon_x, x, mu, logvar, sigma):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / (2 * sigma) + torch.log(sigma)
    # Kullback-Leibler divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

def train_model(
    model,
    train_dataset,
    val_dataset,
    test_dataset,
    max_epochs,
    criterion,
    batch_size,
    lr,
    early_stopping_tolerance=15,
    use_gpu=False
):
    if use_gpu:
        model.cuda()

    early_stopping = EarlyStopping(n_epochs_tolerance=early_stopping_tolerance)

    # Definición de dataloader
    
    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, pin_memory=use_gpu)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, pin_memory=use_gpu)
    
    # Optimizador
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Listas para guardar curvas de entrenamiento
    curves = {
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": []
    }

    t0 = time.perf_counter()

    for epoch in range(max_epochs):
        cumulative_train_loss = 0
        cumulative_train_corrects = 0

        # Entrenamiento del modelo
        model.train()
        for i, (img, temp, diff, y_batch) in enumerate(train_loader):
            # print('\r{}% complete'.format(np.round((epoch + 1)/(max_epochs)*100, decimals = 2)), end='')
            
            if use_gpu:
                img = img.cuda()
                temp = temp.cuda()
                diff = diff.cuda()
                y_batch = y_batch.cuda()
                
            x_combined = torch.cat((img, temp, diff), dim=1)

            # Predicción
            y_predicted, mu, logvar, sigma = model(x_combined)

            y_batch = y_batch.reshape(-1, 1).float()

            # Cálculo de loss

            loss = criterion(y_predicted, y_batch, mu, logvar, sigma)

            # Actualización de parámetros
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cumulative_train_loss += loss.item()

            # Calculamos número de aciertos
            class_prediction = (y_predicted > 0.5)
            cumulative_train_corrects += (y_batch == class_prediction).sum()

        train_loss = cumulative_train_loss / len(train_loader)
        train_acc = cumulative_train_corrects / len(train_dataset)

        # Evaluación del modelo
        model.eval()
        img_val, temp_val, diff_val, y_val = next(iter(val_loader)) #implementar test loader evaluation
        if use_gpu:
            img_val, temp_val, diff_val = img_val.cuda(), temp_val.cuda(), diff_val.cuda()
            y_val = y_val.cuda()

        x_combined_val = torch.cat((img_val, temp_val, diff_val), dim=1)        
        
        y_predicted = model(img_val, temp_val, diff_val)
        y_val = y_val.reshape(-1, 1).float()
        val_loss = criterion(y_predicted, y_val).item()

        class_prediction = (y_predicted > 0.5).long()
        val_acc = (y_val == class_prediction).sum() / y_val.shape[0]

        curves["train_acc"].append(train_acc.item())
        curves["val_acc"].append(val_acc.item())
        curves["train_loss"].append(train_loss)
        curves["val_loss"].append(val_loss)

        print(f'\rEpoch {epoch + 1}/{max_epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}', end='')
        
        if early_stopping(val_loss):
            print(f'\rEpoch {epoch + 1}/{max_epochs} (Early Stop) - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}', end='')
            break

    tiempo_ejecucion = time.perf_counter() - t0
    # print(f"Tiempo total de entrenamiento: {time.perf_counter() - t0:.4f} [s]")

    model.cpu()

    return curves, tiempo_ejecucion