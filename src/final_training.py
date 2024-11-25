import time

import numpy as np

import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import torchvision.transforms.functional as F


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

def augment_data(image, y_batch):

    rotated_90 = torch.stack([F.rotate(img, 90) for img in image])
    rotated_180 = torch.stack([F.rotate(img, 180) for img in image])
    rotated_270 = torch.stack([F.rotate(img, 270) for img in image])

    h_flip = torch.stack([F.hflip(img) for img in image])
    v_flip = torch.stack([F.vflip(img) for img in image])

    imgs = torch.cat((image, rotated_90, rotated_180, rotated_270, h_flip, v_flip), dim=0)
    labels = torch.cat((y_batch, y_batch, y_batch, y_batch, y_batch, y_batch), dim=0)
    
    # Randomize both tensors in the same way so the labels correspond to the images
    indices = torch.randperm(imgs.size(0))
    imgs = imgs[indices]
    labels = labels[indices]

    return imgs, labels

def train_final_model(model,
                      stage,
                      train_dataset,
                      validation_dataset,
                      ae_criterion,
                      rnn_criterion,
                      max_epochs,
                      max_time,
                      batch_size,
                      learning_rate,
                      random_sampler = True,
                      only_classifier = True,
                      augmentation = True,
                      early_stop = True,
                      use_gpu = True,
                      num_cpu = 0):

    #setup

    if stage=='ae_alone':
        pass

    elif stage=='ae_rnn':
        for name, param in model.named_parameters():
            if 'fc3' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    elif stage=='rnn':
        if only_classifier:
            for name, param in model.named_parameters():
                if 'fc3' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:
            for name, param in model.named_parameters():
                param.requires_grad = True
                

    for name, param in model.named_parameters():
        print(f'{name}: {param.requires_grad}')

    if use_gpu:
        model = model.cuda()

    if early_stop != False:
        early_stopping = EarlyStopping(early_stop)

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

            y_batch = y_batch.long()

            if augmentation:
                img, y_batch = augment_data(img, y_batch)

            if use_gpu:
                img, y_batch = img.cuda(), y_batch.cuda()

            if stage == 'ae_alone':
                reconstruction = model.reconstruction(img)
                loss = ae_criterion(reconstruction, img.view(-1, model.n_channels, 21, 21))

            elif stage == 'ae_rnn':
                reconstruction = model.rnn_encode(img)
                loss = ae_criterion(reconstruction, img.view(-1, model.n_channels, 21, 21))

            elif stage == 'rnn':
                prediction = model.rnn_classifier(img)
                loss = rnn_criterion(prediction, y_batch)

            else:
                raise ValueError('Invalid stage value. Pick between "ae_alone", "ae_rnn" or "rnn".')


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

                    print(f"\r({((time.perf_counter() - t_i)/60):.2f} min) Epoch {epoch + 1}/{max_epochs} -- Iteration {iteration} - Batch {i}/{len(train_loader)} - Train loss: {train_loss:.8f} - Val loss: {val_loss:.8f}", end='')

            iteration += 1

        train_loss = cumulative_train_loss / train_loss_count

        #evaluation
        model.eval()
        with torch.no_grad():
            img_val, y_val = next(iter(validation_loader))
            if use_gpu:
                img_val, y_val = img_val.cuda(), y_val.cuda()

            y_val = y_val.long()

            if stage == 'ae_alone':
                reconstruction_val = model.reconstruction(img_val)
                val_loss = ae_criterion(reconstruction_val, img_val.view(-1, model.n_channels, 21, 21))

            elif stage == 'ae_rnn':
                reconstruction_val = model.rnn_encode(img_val)
                val_loss = ae_criterion(reconstruction_val, img_val.view(-1, model.n_channels, 21, 21))

            elif stage == 'rnn':
                prediction_val = model.rnn_classifier(img_val)
                val_loss = rnn_criterion(prediction_val, y_val)

            else:
                raise ValueError('Invalid stage value. Pick between "ae" and "rnn".')


        curves["train_loss"].append(train_loss)
        curves["val_loss"].append(val_loss)

        print(f"\r({((time.perf_counter() - t_i)/60):.2f} min) Epoch {epoch + 1}/{max_epochs} -- Iteration {iteration} - Batch {i}/{len(train_loader)} - Train loss: {loss.item():.8f} - Val loss: {val_loss:.8f}", end='')

        if early_stop != False:
            if early_stopping(val_loss):
                print(f'\r({((time.perf_counter() - t_i)/60):.2f} min) Epoch {epoch + 1}/{max_epochs} (Early Stop) -- Iteration {iteration} - Batch {i}/{len(train_loader)} - Train loss: {loss.item():.8f} - Val loss: {val_loss:.8f}', end='')
                break

        if (time.perf_counter() - t_i)/60 > max_time:
            print(f'\r({((time.perf_counter() - t_i)/60):.2f} min) Epoch {epoch + 1}/{max_epochs} (Max time) -- Iteration {iteration} - Batch {i}/{len(train_loader)} - Train loss: {loss.item():.8f} - Val loss: {val_loss:.8f}', end='')
            break

    model.cpu()
    
    return curves