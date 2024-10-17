import importlib
from torch import nn
import time
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class LinearClassifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(latent_dim, num_classes)
        
    def forward(self, z):
        out = self.linear(z)
        return out

def get_latent_features(models, train_dataset, val_dataset, test_dataset):

    train = np.empty(len(models), dtype=object)
    val = np.empty(len(models), dtype=object)
    test = np.empty(len(models), dtype=object)

    for i, model in enumerate(models):
        model.eval()

        temp_train = TensorDataset(model.encoder(train_dataset.tensors[0]).detach(), train_dataset.tensors[1].detach())
        temp_val = TensorDataset(model.encoder(val_dataset.tensors[0]).detach(), val_dataset.tensors[1].detach())
        temp_test = TensorDataset(model.encoder(test_dataset.tensors[0]).detach(), test_dataset.tensors[1].detach())

        train[i] = temp_train
        val[i] = temp_val
        test[i] = temp_test

    return train, val, test


def train_model(
    model,
    train_dataset,
    val_dataset,
    test_dataset,
    max_epochs,
    criterion,
    batch_size,
    lr,
    use_gpu=False
):
    if use_gpu:
        model.cuda()

    # Definición de dataloader
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=use_gpu)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=0, pin_memory=use_gpu)
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
    val_loss = 0
    all_preds = []
    all_labels = []

    for epoch in range(max_epochs):
        cumulative_train_loss = 0
        cumulative_train_corrects = 0
        train_loss_count = 0
        train_acc_count = 0
        
        # Entrenamiento del modelo
        model.train()
        for i, (batch_features, batch_labels) in enumerate(train_loader):

            if use_gpu:
                batch_features = batch_features.cuda()
                batch_labels = batch_labels.cuda()
            batch_labels = batch_labels.long()
            # Predicción
            reconstruction = model(batch_features)

            # Calcular la pérdida de clasificación
            loss = criterion(reconstruction, batch_labels)
            
            # Actualización de parámetros
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Acumulación de pérdida y conteo de ejemplos procesados
            cumulative_train_loss += loss.item()
            train_loss_count += 1
            train_acc_count += reconstruction.shape[0]

            # Mostrar el progreso
            if i > 0:
                if (i % (len(train_loader) // 100) == 0):  # Si se alcanzó el 1% del total de batches
                    train_loss = cumulative_train_loss / train_loss_count

                    # Aquí `val_loss` debe calcularse previamente en un paso de validación
                    print(f"\rEpoch {epoch + 1}/{max_epochs} -- Iteration {iteration} - Batch {i}/{len(train_loader)} - Train loss: {train_loss:.4f}", end='')

            iteration += 1

            # Calculamos número de aciertos
            # class_prediction = (y_predicted > 0.5)
            # cumulative_train_corrects += (y_batch == class_prediction).sum()

        train_loss = cumulative_train_loss / train_loss_count


        # Evaluación del modelo
        model.eval()  # Cambiar a modo de evaluación
        batch_features, batch_labels = next(iter(val_loader))
        if use_gpu:
            batch_features = batch_features.cuda()
            batch_labels = batch_labels.cuda()
                    
        batch_labels = batch_labels.long()
        # Predicción del modelo
        predictions = model(batch_features)

        # Calcular la pérdida de clasificación (CrossEntropyLoss)
        
        val_loss = criterion(predictions, batch_labels).item()
        _, preds = torch.max(predictions, 1)

        # Almacenar las predicciones y las etiquetas verdaderas
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())
        # class_prediction = (y_predicted > 0.5).long()
        # val_acc = (y_val == class_prediction).sum() / y_val.shape[0]

        # curves["train_acc"].append(train_acc.item())
        # curves["val_acc"].append(val_acc.item())
        curves["train_loss"].append(train_loss)
        curves["val_loss"].append(val_loss)

        print(f"\rEpoch {epoch + 1}/{max_epochs} -- Iteration {iteration} - Batch {i}/{len(train_loader)} - Train loss: {loss.item():4f} - Val loss: {val_loss:.4f}", end='')
        
        # if early_stopping(val_loss):
        #     print(f'\rEpoch {epoch + 1}/{max_epochs} (Early Stop) - Train Loss: {train_loss:.4f} - Val loss: {val_loss:.4f}', end='')
        #     break
    tiempo_ejecucion = time.perf_counter() - t0
    print('\n')
    # print(f"Tiempo total de entrenamiento: {time.perf_counter() - t0:.4f} [s]")

    model.cpu()

    return curves, tiempo_ejecucion, all_labels, all_preds