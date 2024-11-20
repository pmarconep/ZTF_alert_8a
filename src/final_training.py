import time

import numpy as np

import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

class EarlyStopping:
    def __init__(self, n_epochs_tolerance):
        self.n_epochs_tolerance = n_epochs_tolerance
        self.epochs_with_no_improvement = 0
        self.best_loss = np.inf

    def __call__(self, val_loss):
        self.epochs_with_no_improvement += 1

        if val_loss <= self.best_loss:
            self.best_loss = val_loss
            self.epochs_with_no_improvement = 0

        return self.epochs_with_no_improvement >= self.n_epochs_tolerance
    
def augment_data(dataset, batch_size, use_gpu, num_cpu):

    augmented_data = []
    augmented_labels = []

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

        # Horizontal flip
        h_flip_img = torchvision.transforms.functional.hflip(img)
        augmented_data.append(h_flip_img)
        augmented_labels.append(label)

        # Vertical flip
        v_flip_img = torchvision.transforms.functional.vflip(img)
        augmented_data.append(v_flip_img)
        augmented_labels.append(label)

        augmented_dataset = TensorDataset(torch.stack(augmented_data), torch.tensor(augmented_labels))


    return augmented_dataset

def train_final_model(model,
                      train_dataset,
                      validation_dataset,
                      criterion,
                      alpha,
                      max_epochs,
                      max_time,
                      batch_size,
                      learning_rate,
                      random_sampler = True,
                      augmentation = True,
                      early_stop = True,
                      use_gpu = True,
                      num_cpu = 0):

    #setup
    if use_gpu:
        model = model.cuda()

    if early_stop != False:
        early_stopping = EarlyStopping(early_stop)

    if augmentation:
        print('\nAugmenting data...')
        augmented_dataset = augment_data(train_dataset, batch_size, use_gpu, num_cpu)

        if random_sampler:
            class_counts = torch.bincount(augmented_dataset.tensors[1].long())
            class_weights = 1. / class_counts.float()
            weights = class_weights[augmented_dataset.tensors[1].long()]
            
            sampler = WeightedRandomSampler(weights, len(weights))
            
            train_loader = DataLoader(augmented_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_cpu, pin_memory=use_gpu)
        else:
            train_loader = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True, num_workers=num_cpu, pin_memory=use_gpu)
    else:
        if random_sampler:
            class_counts = torch.bincount(train_dataset.tensors[1].long())
            class_weights = 1. / class_counts.float()
            weights = class_weights[train_dataset.tensors[1].long()]
            
            sampler = WeightedRandomSampler(weights, len(weights))
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_cpu, pin_memory=use_gpu)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_cpu, pin_memory=use_gpu)

    validation_loader = DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=False, num_workers=num_cpu, pin_memory=use_gpu)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    curves = {'train_loss': [], 'val_loss': []}
    model_loss = []

    print('\nSetup finishied. Starting training...')

    iteration=0
    n_batches = len(train_loader)
    val_loss = 0

    t_i = time.perf_counter()

    for epoch in range(max_epochs):
        cumulative_train_loss = 0
        train_loss_count = 0

        model.train()
        for i, (img, y_batch) in enumerate(train_loader):

            if use_gpu:
                    img, y_batch = img.cuda(), y_batch.cuda()

            lat_spc = model.encoder(img)
            predictions, mid_lat_spc = model.classifier(lat_spc)
            reconstruction = model.decoder(mid_lat_spc)

            loss = criterion(reconstruction, img, predictions, y_batch, alpha = alpha)

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

                    print(f"\r({((time.perf_counter() - t_i)/60):.2f}s) Epoch {epoch + 1}/{max_epochs} -- Iteration {iteration} - Batch {i}/{len(train_loader)} - Train loss: {train_loss:.8f} - Val loss: {val_loss:.8f}", end='')

            iteration += 1

        train_loss = cumulative_train_loss / train_loss_count

        #evaluation
        model.eval()

        img_val, y_val = next(iter(validation_loader))

        if use_gpu:
            img_val, y_val = img_val.cuda(), y_val.cuda()

        lat_spc_val = model.encoder(img_val)
        predictions_val, mid_lat_spc_val = model.classifier(lat_spc_val)
        reconstruction_val = model.decoder(mid_lat_spc_val)

        val_loss = criterion(reconstruction_val, img_val, predictions_val, y_val, alpha = alpha)      
                   
        curves["train_loss"].append(train_loss)
        curves["val_loss"].append(val_loss)

        print(f"\r({((time.perf_counter() - t_i)/60):.2f}s) Epoch {epoch + 1}/{max_epochs} -- Iteration {iteration} - Batch {i}/{len(train_loader)} - Train loss: {loss.item():.8f} - Val loss: {val_loss:.8f}", end='')

        if early_stop != False:
            if early_stopping(val_loss):
                print(f'\r({((time.perf_counter() - t_i)/60):.2f}s) Epoch {epoch + 1}/{max_epochs} (Early Stop) -- Iteration {iteration} - Batch {i}/{len(train_loader)} - Train loss: {loss.item():.8f} - Val loss: {val_loss:.8f}', end='')
                break

        if (time.perf_counter() - t_i)/60 > max_time:
            print(f'\r({((time.perf_counter() - t_i)/60):.2f}s) Epoch {epoch + 1}/{max_epochs} (Max time) -- Iteration {iteration} - Batch {i}/{len(train_loader)} - Train loss: {loss.item():.8f} - Val loss: {val_loss:.8f}', end='')
            break

    model.cpu()
    
    return curves