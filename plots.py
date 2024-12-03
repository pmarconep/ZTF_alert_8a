#%%
import matplotlib.pyplot as plt
import numpy as np


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
import src.utils.gau as gaussian
import src.utils.plots as plots

from src.models import AE, RNN, LinearClassifier, get_latent_features, rnn_loss_function, ae_loss_function
from src.training import EarlyStopping, train_ae, train_rnn, train_lp, augment_data

from src.final_model import ae_loss_function, rnn_loss_function

#%%
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
#%%

train_dataset.tensors[1]
unique_classes, counts = torch.unique(train_dataset.tensors[1], return_counts=True)
class_counts = dict(zip(unique_classes.tolist(), counts.tolist()))
print(class_counts)


#%%
# plot examples of classes
def plot_examples(dataset, title, t):
    images = dataset.tensors[0]
    labels = dataset.tensors[1]
    class_indices = {0: [], 1: [], 2: []}
    for i, label in enumerate(labels):
        class_indices[int(label.item())].append(i)

    random_indices = {cls: np.random.choice(indices) for cls, indices in class_indices.items()}
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    fig.suptitle(f'{title}')
    print(random_indices)

    axes[0, 0].set_ylabel('AGN')
    axes[1, 0].set_ylabel('SNe')
    axes[2, 0].set_ylabel('VS')

    axes[0, 0].set_title(f'Template')
    axes[0, 1].set_title(f'Science')
    axes[0, 2].set_title(f'Difference')
        
    for j in range(3):
        template = images[random_indices[j]][t][0]
        scienc = images[random_indices[j]][t][1]
        diff = images[random_indices[j]][t][2]
        axes[j, 0].imshow(template, cmap='viridis')
        axes[j, 1].imshow(scienc, cmap='viridis')
        axes[j, 2].imshow(diff, cmap='viridis')
        
        for i in range(3):
            axes[i, j].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    return fig

# fig1 = plot_examples(train_dataset, 'Train Dataset Examples', 0)
# fig2 = plot_examples(validation_dataset, 'Validation Dataset Examples', 0)
# fig3 = plot_examples(test_dataset, 'Test Dataset Examples', 0)

# fig1.savefig('plots/train_dataset_example.png')
# fig2.savefig('plots/validation_dataset_example.png')
# fig3.savefig('plots/test_dataset_example.png')


# %%

#timestamps example

def plot_custom_layout(dataset, title, index):
    images = dataset.tensors[0]
    labels = dataset.tensors[1]

    class_indices = {0: [], 1: [], 2: []}
    for i, label in enumerate(labels):
        class_indices[int(label.item())].append(i)
    random_indices = {cls: np.random.choice(indices) for cls, indices in class_indices.items()}

    fig, axes = plt.subplots(3, 5, figsize=(8, 5))
    fig.suptitle(title)

    axes[0, 0].set_ylabel('AGN')
    axes[1, 0].set_ylabel('SNe')
    axes[2, 0].set_ylabel('VS')

    axes[0, 0].set_title(f't=1')
    axes[0, 1].set_title(f't=2')
    axes[0, 2].set_title(f't=3')
    axes[0, 3].set_title(f't=4')
    axes[0, 4].set_title(f't=5')

    # Plot the other 5 columns with 2 rows
    for j in range(5):
        for i in range(3):
            t1 = images[random_indices[i]][0][1]
            t2 = images[random_indices[i]][1][1]
            t3 = images[random_indices[i]][2][1]
            t4 = images[random_indices[i]][3][1]
            t5 = images[random_indices[i]][4][1]
            axes[i, 0].imshow(t1, cmap='viridis')
            axes[i, 1].imshow(t2, cmap='viridis')
            axes[i, 2].imshow(t3, cmap='viridis')
            axes[i, 3].imshow(t4, cmap='viridis')
            axes[i, 4].imshow(t5, cmap='viridis')
            axes[i, j].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    return fig

# fig4 = plot_custom_layout(train_dataset, 'Train dataset timestamps example', 10000)
# fig5 = plot_custom_layout(validation_dataset, 'Validation dataset timestamps example', 500)
# fig6 = plot_custom_layout(test_dataset, 'Test dataset timestamps example', 500)

# fig4.savefig('plots/train_dataset_timestamps_example.png')
# fig5.savefig('plots/validation_dataset_timestamps_example.png')
# fig6.savefig('plots/test_dataset_timestamps_example.png')

# %%
import src.final_model as fm
import src.final_training as ft
importlib.reload(fm)
importlib.reload(ft)
importlib.reload(src.final_training)
importlib.reload(src.final_model)
importlib.reload(src.utils.metrics)
importlib.reload(metrics)

modelo_final = fm.FinalModel(128, 3, 'RNN', 128, 1, 0.35,  3, name='best model')
modelo_final.load_state_dict(torch.load(f'models/1st_iteration.pth'))
modelo_final.cpu()
modelo_final.eval()

# prediction_train = modelo_final.rnn_classifier_test(validation_dataset.tensors[0])
# prediction_test = modelo_final.rnn_classifier_test(test_dataset.tensors[0])
prediction_validation = modelo_final.rnn_classifier_test(validation_dataset.tensors[0])

# reconstruction_train = modelo_final.rnn_encode_test(validation_dataset.tensors[0])

# reconstruction_test = modelo_final.rnn_encode_test(validation_dataset.tensors[0])

# reconstructions
# %%

def plot_reconstruction(model, dataset, title):
    images = dataset.tensors[0]
    labels = dataset.tensors[1]
    # reconstructions = model.rnn_encode_test(images).view(-1, 5, 3, 21, 21).detach().numpy()
    # print(reconstructions.shape)
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    fig.suptitle(title)

    class_indices = {0: [], 1: [], 2: []}
    for i, label in enumerate(labels):
        class_indices[int(label.item())].append(i)

    random_indices = {cls: np.random.choice(indices) for cls, indices in class_indices.items()}

    axes[0, 0].set_ylabel('AGN')
    axes[1, 0].set_ylabel('SNe')
    axes[2, 0].set_ylabel('VS')

    axes[0, 0].set_title('Original')
    axes[0, 1].set_title('Reconstructed')
    axes[0, 2].set_title('Difference')

    for j in range(3):
        print(images[random_indices[j]].shape)
        original = images[random_indices[j]][0][1]
        reconstructed = model.rnn_encode_test(images[random_indices[j]].view(1, 5, 3, 21, 21))[0][1].detach().numpy()

        # Normalize from 0 to 1
        original = (original - original.min()) / (original.max() - original.min())
        reconstructed = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min())

        difference = original - reconstructed

        axes[j, 0].imshow(original, cmap='viridis')
        axes[j, 1].imshow(reconstructed, cmap='viridis')
        axes[j, 2].imshow(difference, cmap='viridis')

        for i in range(3):
            axes[j, i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    return fig

# fig7 = plot_reconstruction(modelo_final, train_dataset, 'Train dataset reconstructions')
# fig8 = plot_reconstruction(modelo_final, test_dataset, 'Test dataset reconstructions')

# fig7.savefig('plots/validation_dataset_reconstructions.png')
# fig8.savefig('plots/test_dataset_reconstructions.png')

# %%

import umap

def plot_umap(predictions, labels, title):
    umaps = umap.UMAP(n_components=2, random_state=42)
    embeddings = umaps.fit_transform(predictions.detach().numpy())
    classes = ['AGN', 'SNe', 'VS']
    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        indices = np.where(labels == label)
        plt.scatter(embeddings[indices, 0], embeddings[indices, 1], label=f'Class {classes[int(label)]}', alpha=0.7)
    
    plt.title(title)
    plt.legend()

# prediction_test
# Plot UMAP for train and test predictions
# plot_umap(prediction_train, validation_class_0, 'UMAP of best model train predictions')
plot_umap(prediction_validation, validation_class_0, 'UMAP of best model validation predictions')

# plot_umap(prediction_test, test_class_0, 'UMAP of best model test predictions')