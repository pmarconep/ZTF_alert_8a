
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

def augment_data(dataset, batch_size, use_gpu, shuffle = False):
    augmented_data = []
    augmented_labels = []

    print(f'Augmenting Data ... ')
    for i, (img, label) in enumerate(dataset):
        augmented_data.append(img)
        augmented_labels.append(label)

        # Rotate 90 degrees
        img_90 = torchvision.transforms.functional.rotate(img, 90)
        augmented_data.append(img_90)
        augmented_labels.append(label)
        img_180 = torchvision.transforms.functional.rotate(img, 180)
        augmented_data.append(img_180)
        augmented_labels.append(label)
        img_270 = torchvision.transforms.functional.rotate(img, 270)
        augmented_data.append(img_270)
        augmented_labels.append(label)

        #Divide the image into 3x3 squares and randomly shuffle them
        if shuffle:
            img_np = np.array(img.permute(1, 2, 0))  # Convert to numpy array and move channels to last dimension
            h, w, c = img_np.shape
            square_size_h = h // 3
            square_size_w = w // 3

            squares = []
            for i in range(3):
                for j in range(3):
                    square = img_np[i * square_size_h:(i + 1) * square_size_h, j * square_size_w:(j + 1) * square_size_w, :]
                    squares.append(square)

            np.random.shuffle(squares)

            shuffled_img_np = np.zeros_like(img_np)
            for i in range(3):
                for j in range(3):
                    shuffled_img_np[i * square_size_h:(i + 1) * square_size_h, j * square_size_w:(j + 1) * square_size_w, :] = squares[i * 3 + j]

            shuffled_img = torch.tensor(shuffled_img_np).permute(2, 0, 1)  # Convert back to tensor and move channels to first dimension
            augmented_data.append(shuffled_img)
            augmented_labels.append(label)

        h_flip_img = torchvision.transforms.functional.hflip(img)
        augmented_data.append(h_flip_img)
        augmented_labels.append(label)

        # Vertical flip
        v_flip_img = torchvision.transforms.functional.vflip(img)
        augmented_data.append(v_flip_img)
        augmented_labels.append(label)

    augmented_dataset = TensorDataset(torch.stack(augmented_data), torch.tensor(augmented_labels))
    train_loader = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=use_gpu)

    return train_loader


def train_model(
    model,
    train_dataset,
    val_dataset,
    test_dataset,
    max_epochs,
    criterion,
    batch_size,
    lr,
    augmentation = False,
    shuffle_augmentation = False,
    early_stop = False,
    use_gpu=False,
):
    #setup
    if use_gpu:
        model.cuda()

    if early_stop != False:
        early_stopping = EarlyStopping(early_stop)

    if augmentation:
        train_loader = augment_data(train_dataset, batch_size, use_gpu, shuffle = shuffle_augmentation)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=use_gpu)

    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=0, pin_memory=use_gpu)  
  
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    curves = {
        "train_loss": [],
        "val_loss": []
    }

    model_loss = []
    #start training
    print('Starting training ...')

    iteration = 0
    n_batches = len(train_loader)
    val_loss = 0

    t0 = time.perf_counter()

    for epoch in range(max_epochs):
        cumulative_train_loss = 0
        train_loss_count = 0
     
        # Entrenamiento del modelo
        model.train()
        for i, (diff, y_batch) in enumerate(train_loader):
            if use_gpu:
                diff = diff.cuda()
            print(diff.shape)

            reconstruction = model(diff)

            loss = criterion(reconstruction, diff).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cumulative_train_loss += loss.item()
            train_loss_count += 1
            
            
            batch_losses = loss.item() 
            model_loss.append(batch_losses) 
            
            if i > 0:
                if (i % (n_batches // 100) == 0):
                    train_loss = cumulative_train_loss / train_loss_count

                    print(f"\rEpoch {epoch + 1}/{max_epochs} -- Iteration {iteration} - Batch {i}/{len(train_loader)} - Train loss: {train_loss:.8f} - Val loss: {val_loss:.8f}", end='')

            iteration += 1

        train_loss = cumulative_train_loss / train_loss_count
        
        # Evaluación del modelo
        model.eval()
        diff_val, y_val = next(iter(val_loader))
        if use_gpu:
            diff_val = diff_val.cuda()
            y_val = y_val.cuda()        
        
        reconstruction = model(diff_val)
           
        val_loss = criterion(reconstruction, diff_val).mean().item()

        curves["train_loss"].append(train_loss)
        curves["val_loss"].append(val_loss)

        print(f"\rEpoch {epoch + 1}/{max_epochs} -- Iteration {iteration} - Batch {i}/{len(train_loader)} - Train loss: {loss.item():.8f} - Val loss: {val_loss:.8f}", end='')

        if early_stop != False:
            if early_stopping(val_loss):
                print(f'\rEpoch {epoch + 1}/{max_epochs} (Early Stop) -- Iteration {iteration} - Batch {i}/{len(train_loader)} - Train loss: {loss.item():.8f} - Val loss: {val_loss:.8f}', end='')
                break

    tiempo_ejecucion = time.perf_counter() - t0
    print(f"Tiempo total de entrenamiento: {time.perf_counter() - t0:.2f} [s]\n")
    
    total_mse_loss = np.mean(np.array(model_loss))
    model.cpu()

    return curves, tiempo_ejecucion, total_mse_loss