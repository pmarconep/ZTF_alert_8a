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
#dataset_1_21 = pd.read_pickle('data/stamp_dataset_21_new.pkl')

# %%
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

# Verificar las dimensiones de los conjuntos de datos
print(train_dataset.tensors[0].shape)  # (samples, 5, 3, 21, 21)
print(validation_dataset.tensors[0].shape)
print(test_dataset.tensors[0].shape)



# %%
importlib.reload(train_recurrent)
importlib.reload(rnn)
importlib.reload(ae)

model = ae.AE(50, 3, name="Test_recurrent")
model.load_state_dict(torch.load('models/model_final.pth'))
model.eval()


hidden_dim = 5
num_layers = 1

rnn_model = rnn.RNN(5, 50, hidden_dim, num_layers, 3, name = "RNN_test")

curves, time_i = train_recurrent.train_model(model,
                                             rnn_model,
                                             train_dataset,
                                             validation_dataset,
                                             test_dataset, 
                                             max_epochs=100, 
                                             batch_size=10,
                                             lr=1e-3, 
                                             early_stop=5,
                                             criterion=rnn.loss_function
                                             )

torch.save(rnn_model.state_dict(), 'models/rnn_model_final.pth')
# %%



