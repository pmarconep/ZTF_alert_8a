
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib

#system
from pathlib import Path
import time

#ai
import torch
from torch import nn
import torchvision
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import src
importlib.reload(src)

import src.model.train as train
import src.model.ae as ae

import src.probing.linear_probing as lp

import src.utils.plots as plots
import src.utils.gau as gaussian
import src.utils.metrics as metrics

stamps_dataset = pd.read_pickle('data/5stamps_dataset.pkl')

def rename_labels(dataset, old_value, new_value):
    for key in dataset.keys():
        if old_value in dataset[key]:
            dataset[key][new_value] = dataset[key].pop(old_value)

rename_labels(stamps_dataset, 'labels', 'class')
rename_labels(stamps_dataset, 'science', 'images')

train_template = torch.tensor(stamps_dataset['Train']['template'], dtype=torch.float32)
validation_template = torch.tensor(stamps_dataset['Validation']['template'], dtype=torch.float32)
test_template = torch.tensor(stamps_dataset['Test']['template'], dtype=torch.float32)

train_difference = torch.tensor(stamps_dataset['Train']['difference'], dtype=torch.float32)
validation_difference = torch.tensor(stamps_dataset['Validation']['difference'], dtype=torch.float32)
test_difference = torch.tensor(stamps_dataset['Test']['difference'], dtype=torch.float32)

train_image = torch.tensor(stamps_dataset['Train']['images'], dtype=torch.float32)
validation_image = torch.tensor(stamps_dataset['Validation']['images'], dtype=torch.float32)
test_image = torch.tensor(stamps_dataset['Test']['images'], dtype=torch.float32)

train_class_0 = torch.tensor(stamps_dataset['Train']['class'], dtype=torch.float32)
validation_class_0 = torch.tensor(stamps_dataset['Validation']['class'], dtype=torch.float32)
test_class_0 = torch.tensor(stamps_dataset['Test']['class'], dtype=torch.float32)

# Reshape the arrays to separate timestamps
num_samples, num_photos, height, width = train_difference.shape
reshaped_train_difference = train_difference.reshape(num_samples * num_photos, height, width)
reshaped_train_template = train_template.repeat_interleave(num_photos, dim=0)
reshaped_train_image = train_image.reshape(train_image.shape[0] * num_photos, height, width)
reshaped_train_class = train_class_0.repeat_interleave(num_photos)

num_samples, num_photos, height, width = validation_difference.shape
reshaped_val_difference = validation_difference.reshape(validation_difference.shape[0] * num_photos, height, width)
reshaped_val_template = validation_template.repeat_interleave(num_photos, dim=0)
reshaped_val_image = validation_image.reshape(validation_image.shape[0] * num_photos, height, width)
reshaped_val_class = validation_class_0.repeat_interleave(num_photos)

num_samples, num_photos, height, width = test_difference.shape
reshaped_test_difference = test_difference.reshape(test_difference.shape[0] * num_photos, height, width)
reshaped_test_template = test_template.repeat_interleave(num_photos, dim=0)
reshaped_test_image = test_image.reshape(test_image.shape[0] * num_photos, height, width)
reshaped_test_class = test_class_0.repeat_interleave(num_photos)

# Combine the template and difference into a 2-channel image
train_dataset = torch.stack((reshaped_train_template, reshaped_train_image, reshaped_train_difference), dim=3)
validation_dataset = torch.stack((reshaped_val_template, reshaped_val_image, reshaped_val_difference), dim=3)
test_dataset = torch.stack((reshaped_test_template, reshaped_test_image, reshaped_test_difference), dim=3)

train_dataset_0 = TensorDataset(train_dataset.permute(0, 3, 1, 2), reshaped_train_class)
validation_dataset_0 = TensorDataset(validation_dataset.permute(0, 3, 1, 2), reshaped_val_class)
test_dataset_0 = TensorDataset(test_dataset.permute(0, 3, 1, 2), reshaped_test_class)

from torch.utils.data import random_split

first = int(len(train_dataset_0)/4)
second = int(len(train_dataset_0)/4)
third = int(len(train_dataset_0)/4)
fourth = len(train_dataset_0) - (first + second + third)

train_subset_0_0, train_subset_0_1, train_subset_0_2, train_subset_0_3 = random_split(train_dataset_0, [first, second, third, fourth], generator=torch.Generator().manual_seed(42))

lp_epochs = 200
lp_criterion = nn.CrossEntropyLoss()
lp_batch_size = 100
lp_lr = 1.665e-4
use_gpu = True

import src.probing.train as train_lp

# %%
first_half_indexs = train_subset_0_0.indices
second_half_indexs = train_subset_0_1.indices
third_half_indexs = train_subset_0_2.indices
fourth_half_indexs = train_subset_0_3.indices

indexes = np.array([first_half_indexs, second_half_indexs, third_half_indexs, fourth_half_indexs], dtype=object)
np.save('indexes.npy', indexes)

# %%
loaded_indexes = np.load('indexes.npy', allow_pickle=True)
first_half = TensorDataset(train_dataset_0.tensors[0][loaded_indexes[0]], train_dataset_0.tensors[1][loaded_indexes[0]])
second_half = TensorDataset(train_dataset_0.tensors[0][loaded_indexes[1]], train_dataset_0.tensors[1][loaded_indexes[1]])
third_half = TensorDataset(train_dataset_0.tensors[0][loaded_indexes[2]], train_dataset_0.tensors[1][loaded_indexes[2]])
fourth_half = TensorDataset(train_dataset_0.tensors[0][loaded_indexes[3]], train_dataset_0.tensors[1][loaded_indexes[3]])

model = ae.AE(latent_dim = 50, n_channels = 3, name = 'final')
model.load_state_dict(torch.load('models/model_final.pth'))

lp_model_0= lp.LinearClassifier(42, 3, 'final_database')

train_0, val_0, test_0 = lp.get_latent_features([model], first_half, validation_dataset_0, test_dataset_0)

lp_curves_1, lp_tiempo_1, lp_labels_1, lp_pred_1 = train_lp.train_model(lp_model_0, train_0[0], val_0[0], test_0[0], lp_epochs, lp_criterion, lp_batch_size, lp_lr, use_gpu)

del train_0

torch.save(lp_model_0.state_dict(), 'models/lp_model_final.pth')

train_1, val_1, test_1 = lp.get_latent_features([model], second_half, validation_dataset_0, test_dataset_0)

lp_curves_1, lp_tiempo_1, lp_labels_1, lp_pred_1 = train_lp.train_model(lp_model_0, train_1[0], val_1[0], test_1[0], lp_epochs, lp_criterion, lp_batch_size, lp_lr, use_gpu)

del train_1

torch.save(lp_model_0.state_dict(), 'models/lp_model_final.pth')

train_2, val_2, test_2= lp.get_latent_features([model], third_half, validation_dataset_0, test_dataset_0)

lp_curves_1, lp_tiempo_1, lp_labels_1, lp_pred_1 = train_lp.train_model(lp_model_0, train_1[0], val_1[0], test_1[0], lp_epochs, lp_criterion, lp_batch_size, lp_lr, use_gpu)

del train_2

torch.save(lp_model_0.state_dict(), 'models/lp_model_final.pth')

train_3, val_3, test_3 = lp.get_latent_features([model], fourth_half, validation_dataset_0, test_dataset_0)

lp_curves_1, lp_tiempo_1, lp_labels_1, lp_pred_1 = train_lp.train_model(lp_model_0, train_1[0], val_1[0], test_1[0], lp_epochs, lp_criterion, lp_batch_size, lp_lr, use_gpu)

del train_3

torch.save(lp_model_0.state_dict(), 'models/lp_model_final.pth')

# trains = np.array([train_0, train_1, train_2, train_3], dtype=object) 
vals = np.array([val_0, val_1, val_2, val_3], dtype=object)
tests = np.array([test_0, test_1, test_2, test_3], dtype=object)

np.save('vals.npy', vals)
np.save('tests.npy', tests)