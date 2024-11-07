import importlib
from torch import nn
import time
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.nn.functional as F


class LinearClassifier(nn.Module):
    def __init__(self, latent_dim, num_classes, name):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(latent_dim, num_classes)
        self.name = name
        
    def forward(self, z):
        out = self.linear(z)
        return out

def get_latent_features(models, train_dataset, val_dataset, test_dataset):

    model = models[0]
    
    train = np.empty(len(models), dtype=object)
    val = np.empty(len(models), dtype=object)
    test = np.empty(len(models), dtype=object)

    train = TensorDataset(model.encoder(train_dataset.tensors[0]).detach(), train_dataset.tensors[1].detach())
    val = TensorDataset(model.encoder(val_dataset.tensors[0]).detach(), val_dataset.tensors[1].detach())
    test = TensorDataset(model.encoder(test_dataset.tensors[0]).detach(), test_dataset.tensors[1].detach())

    return train, val, test