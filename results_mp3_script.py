# %%
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

import src.utils.metrics as metrics
import src.model.train_rnn as train_recurrent
import src.model.ae as ae
import src.model.rnn as rnn
import src.utils.gau as gaussian
import src.probing.linear_probing as lp
import src.utils.plots as plots

# %%
stamps_dataset = pd.read_pickle('data/5stamps_dataset.pkl')

def rename_labels(dataset, old_value, new_value):
    for key in dataset.keys():
        if old_value in dataset[key]:
            dataset[key][new_value] = dataset[key].pop(old_value)

rename_labels(stamps_dataset, 'labels', 'class')
rename_labels(stamps_dataset, 'science', 'images')

# %%
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

train_template = train_template.unsqueeze(1).repeat(1, 5, 1, 1)
validation_template = validation_template.unsqueeze(1).repeat(1, 5, 1, 1)
test_template = test_template.unsqueeze(1).repeat(1, 5, 1, 1)


train_dataset = torch.stack((train_template, train_image, train_difference), dim=3  )
validation_dataset = torch.stack((validation_template, validation_difference, validation_difference), dim=3)
test_dataset = torch.stack((test_template, test_image, test_difference), dim=3)

train_template = train_template.unsqueeze(2)  # (samples, 5, 1, 21, 21)
train_image = train_image.unsqueeze(2)        
train_difference = train_difference.unsqueeze(2)  

validation_template = validation_template.unsqueeze(2)
validation_image = validation_image.unsqueeze(2)
validation_difference = validation_difference.unsqueeze(2)

test_template = test_template.unsqueeze(2)
test_image = test_image.unsqueeze(2)
test_difference = test_difference.unsqueeze(2)

# Apilar los tensores a lo largo de la dimensión correcta
train_dataset = torch.cat((train_template, train_image, train_difference), dim=2)
validation_dataset = torch.cat((validation_template, validation_image, validation_difference), dim=2)
test_dataset = torch.cat((test_template, test_image, test_difference), dim=2)

# Crear los conjuntos de datos
train_dataset = TensorDataset(train_dataset, train_class_0)
validation_dataset = TensorDataset(validation_dataset, validation_class_0)
test_dataset = TensorDataset(test_dataset, test_class_0)

train_dataset.tensors[1]
unique, counts = torch.unique(train_dataset.tensors[1], return_counts=True)
print(dict(zip(unique.numpy(), counts.numpy())))

from torch.utils.data import Subset

# Get the indices of each class
class_indices = {cls: (train_dataset.tensors[1] == cls).nonzero(as_tuple=True)[0] for cls in unique}

# Find the minimum number of samples in any class
min_samples = min(len(indices) for indices in class_indices.values())

# Create balanced indices by sampling min_samples from each class
balanced_indices = torch.cat([indices[:min_samples] for indices in class_indices.values()])

# Create a balanced dataset
balanced_train_dataset = Subset(train_dataset, balanced_indices)

# Verify the balance
balanced_counts = torch.unique(balanced_train_dataset.dataset.tensors[1][balanced_indices], return_counts=True)

# %%
model = ae.AE(50, 3, name="autoencoder final")
model.load_state_dict(torch.load('models/model_final.pth'))
model.eval()

# %%
hidden_dim = 64
num_layers = 2
lr = 0.666e-3
batch_size = 1
use_gpu = False

# %% [markdown]
# # Experimentos Distintos Modelos

# %%
# importlib.reload(rnn)
# importlib.reload(train_recurrent)
# importlib.reload(metrics)

# rnn_model_rnn = rnn.RNN('RNN', 5, 50, hidden_dim, num_layers, 3, name = "RNN")

# curves_1, time_1 = train_recurrent.train_model(model,
#                                              rnn_model_rnn,
#                                              balanced_train_dataset,
#                                              validation_dataset,
#                                              test_dataset, 
#                                              max_epochs=100, 
#                                              batch_size=batch_size,
#                                              lr=lr, 
#                                              early_stop=6,
#                                              criterion=rnn.loss_function,
#                                              use_gpu=use_gpu
#                                              )

# torch.save(rnn_model_rnn.state_dict(), 'models/rnn_rnn_model_final.pth')

# # %%
# importlib.reload(rnn)
# importlib.reload(train_recurrent)
# importlib.reload(metrics)

# rnn_model_lstm = rnn.RNN('LSTM', 5, 50, hidden_dim, num_layers, 3, name = "LSTM")

# curves_2, time_2 = train_recurrent.train_model(model,
#                                              rnn_model_lstm,
#                                              balanced_train_dataset,
#                                              validation_dataset,
#                                              test_dataset, 
#                                              max_epochs=100, 
#                                              batch_size=batch_size,
#                                              lr=lr, 
#                                              early_stop=6,
#                                              criterion=rnn.loss_function,
#                                              use_gpu=use_gpu
#                                              )

# torch.save(rnn_model_lstm.state_dict(), 'models/rnn_lstm_model_final.pth')


# # %%
# importlib.reload(rnn)
# importlib.reload(train_recurrent)
# importlib.reload(metrics)

# rnn_model_gru = rnn.RNN('GRU', 5, 50, hidden_dim, num_layers, 3, name = "GRU")

# curves_3, time_3 = train_recurrent.train_model(model,
#                                                 rnn_model_gru,
#                                                 balanced_train_dataset,
#                                                 validation_dataset,
#                                                 test_dataset, 
#                                                 max_epochs=100, 
#                                                 batch_size=batch_size,
#                                                 lr=lr, 
#                                                 early_stop=6,
#                                                 criterion=rnn.loss_function,
#                                                 use_gpu=use_gpu
#                                                 )

# torch.save(rnn_model_gru.state_dict(), 'models/rnn_gru_model_final.pth')

# %% [markdown]
# # Augmentation

# %%
batch_size = 10

# %% [markdown]
# ## Class Weights

# %%
importlib.reload(rnn)
importlib.reload(train_recurrent)
importlib.reload(metrics)

class_counts = torch.bincount(train_dataset.tensors[1].long())
class_weights = 1. / class_counts.float()
weights = class_weights[train_dataset.tensors[1].long()]

rnn_model_cw = rnn.RNN('RNN', 5, 50, hidden_dim, num_layers, 3, name = "LSTM with ClassWeights")

curves_4, time_4 = train_recurrent.train_model(model,
                                                rnn_model_cw,
                                                train_dataset,
                                                validation_dataset,
                                                test_dataset, 
                                                max_epochs=100, 
                                                batch_size=batch_size,
                                                lr=lr, 
                                                early_stop=6,
                                                weights=weights,
                                                criterion=rnn.loss_function,
                                                use_gpu=use_gpu
                                                )

torch.save(rnn_model_cw.state_dict(), 'models/rnn_cw_model_final.pth')

# %% [markdown]
# ## Weighted Random Sampler

# %%
importlib.reload(rnn)
importlib.reload(train_recurrent)
importlib.reload(metrics)

rnn_model_rs = rnn.RNN('RNN', 5, 50, hidden_dim, num_layers, 3, name = "LSTM with ClassWeights")

curves_5, time_5 = train_recurrent.train_model(model,
                                                rnn_model_rs,
                                                train_dataset,
                                                validation_dataset,
                                                test_dataset, 
                                                max_epochs=100, 
                                                batch_size=batch_size,
                                                lr=lr, 
                                                early_stop=6,
                                                weights=False,
                                                random_sampler = True,
                                                criterion=rnn.loss_function,
                                                use_gpu=use_gpu
                                                )

torch.save(rnn_model_rs.state_dict(), 'models/rnn_rs_model_final.pth')

# %% [markdown]
# # Results

# %%
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

latent_test = model.time_sequence(test_dataset.tensors[0])
real_class = test_dataset.tensors[1]

# %% [markdown]
# ## RNN, LTSM, GRU

# %%
#RNN
# predicted = torch.argmax(F.softmax(rnn_model_rnn(latent_test), dim=1), dim=1)

# precision, recall, f1 = metrics.performance_metrics(real_class, predicted)
# cm = confusion_matrix(real_class, predicted)

# print(f"Model: RNN",
#       f"\nPrecision: {precision}",
#       f"\nRecall: {recall}",
#       f"\nF1: {f1}\n")

# fig1 = metrics.plot_matrix([rnn_model_rnn], [cm])
# fig1.savefig('figures/cm_rnn.png')

# # %%
# #LTSM
# predicted = torch.argmax(F.softmax(rnn_model_lstm(latent_test), dim=1), dim=1)

# precision, recall, f1 = metrics.performance_metrics(real_class, predicted)
# cm = confusion_matrix(real_class, predicted)

# print(f"Model: LSTM",
#       f"\nPrecision: {precision}",
#       f"\nRecall: {recall}",
#       f"\nF1: {f1}\n")

# fig2 = metrics.plot_matrix([rnn_model_lstm], [cm])
# fig2.savefig('figures/cm_lstm.png')

# # %%
# #GRU
# predicted = torch.argmax(F.softmax(rnn_model_gru(latent_test), dim=1), dim=1)

# precision, recall, f1 = metrics.performance_metrics(real_class, predicted)
# cm = confusion_matrix(real_class, predicted)

# print(f"Model: GRU",
#       f"\nPrecision: {precision}",
#       f"\nRecall: {recall}",
#       f"\nF1: {f1}\n")

# fig3 = metrics.plot_matrix([rnn_model_gru], [cm])
# fig3.savefig('figures/cm_gru.png')

# %% [markdown]
# ## Augmentations

# %%
#Weight Class
predicted = torch.argmax(F.softmax(rnn_model_cw(latent_test), dim=1), dim=1)

precision, recall, f1 = metrics.performance_metrics(real_class, predicted)
cm = confusion_matrix(real_class, predicted)

print(f"Model: Weight Class",
      f"\nPrecision: {precision}",
      f"\nRecall: {recall}",
      f"\nF1: {f1}\n")

fig4 = metrics.plot_matrix([rnn_model_cw], [cm])
fig4.savefig('figures/cm_cw.png')

# %%
#Random Sampler
predicted = torch.argmax(F.softmax(rnn_model_rs(latent_test), dim=1), dim=1)

precision, recall, f1 = metrics.performance_metrics(real_class, predicted)
cm = confusion_matrix(real_class, predicted)

print(f"Model: Random Sampler",
      f"\nPrecision: {precision}",
      f"\nRecall: {recall}",
      f"\nF1: {f1}\n")

fig5 = metrics.plot_matrix([rnn_model_rs], [cm])
fig5.savefig('figures/cm_rs.png')


