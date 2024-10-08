
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=use_gpu)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, pin_memory=use_gpu)
    # test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, pin_memory=use_gpu)
    
    # Optimizador
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Listas para guardar curvas de entrenamiento
    curves = {
        "train_loss": [],
        "val_loss": []
    }

    t0 = time.perf_counter()
    
    iteration = 0
    n_batches = len(train_loader)

    for epoch in range(max_epochs):
        cumulative_train_loss = 0
        cumulative_train_corrects = 0
        print(f'\rTraining model: epoch {epoch} out of {max_epochs}', end='')
        # Entrenamiento del modelo
        model.train()
        for i, (diff, y_batch) in enumerate(train_loader):
            # diff = diff#.unsqueeze(1).permute(1,0,2,3)
            #print(diff.shape, y_batch.shape)
            if use_gpu:
                diff = diff.cuda()
                # y_batch = y_batch.cuda()

            # Predicción
            reconstruction, mu, logvar, sigma = model(diff)
           
            loss = criterion(reconstruction, diff, mu, logvar, sigma).mean()
            # Actualización de parámetros
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cumulative_train_loss += loss.item()
            
            if (i % (n_batches // 6) == 0) and (i > 0):

                print(f"\rEpoch {epoch + 1}/{max_epochs} -- Iteration {iteration} - Batch {i}/{len(train_loader)} - Train loss: {train_loss:4f}, Train acc: {train_acc:4f}", end='')

            iteration += 1

            # Calculamos número de aciertos
            # class_prediction = (y_predicted > 0.5)
            # cumulative_train_corrects += (y_batch == class_prediction).sum()

        train_loss = cumulative_train_loss / len(train_loader)
        # train_acc = cumulative_train_corrects / len(train_dataset)

        # Evaluación del modelo
        model.eval()
        diff_val, y_val = next(iter(val_loader)) #implementar test loader evaluation
        if use_gpu:
            diff_val = diff_val.cuda()
            y_val = y_val.cuda()        
        
        reconstruction, mu, logvar, sigma = model(diff)
           
        val_loss = criterion(reconstruction, diff, mu, logvar, sigma).mean().item()


        # class_prediction = (y_predicted > 0.5).long()
        # val_acc = (y_val == class_prediction).sum() / y_val.shape[0]

        # curves["train_acc"].append(train_acc.item())
        # curves["val_acc"].append(val_acc.item())
        curves["train_loss"].append(train_loss)
        curves["val_loss"].append(val_loss)

        # print(f'\rEpoch {epoch + 1}/{max_epochs} - Train Loss: {train_loss:.4f} - Val loss: {val_loss:.4f}', end='')
        
        # if early_stopping(val_loss):
        #     print(f'\rEpoch {epoch + 1}/{max_epochs} (Early Stop) - Train Loss: {train_loss:.4f} - Val loss: {val_loss:.4f}', end='')
        #     break

    tiempo_ejecucion = time.perf_counter() - t0
    print('\n')
    # print(f"Tiempo total de entrenamiento: {time.perf_counter() - t0:.4f} [s]")

    model.cpu()

    return curves, tiempo_ejecucion