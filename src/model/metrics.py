import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import torch

def performance_metrics(true_labels, prediction):
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, prediction, average= "macro")
    precision = round(float(precision), 5)
    recall = round(float(recall), 5)
    f1 = round(float(f1), 5)
    return precision, recall, f1

def show_curves(curves, models):
    fig, ax = plt.subplots(1, len(curves), figsize=((13/2)*len(curves), 5), dpi = 300)
    fig.set_facecolor('white')

    epochs = np.arange(len(curves["val_loss"])) + 1

    for i, curve in enumerate(curves):
        ax[i].plot(epochs, curve['val_loss'], label='validation')
        ax[i].plot(epochs, curve['train_loss'], label='training')
        ax[i].set_xlabel('Epoch')
        ax[i].set_ylabel('Loss')
        ax[i].set_title(f'Loss evolution during training model {models[i].name}')
        ax[i].legend()

    plt.show()
    return fig

import umap

def plot_umap(models, data, labels, n_neighbors, min_dist, metric, norm = True):

    fig, ax = plt.subplots(1, len(models), figsize=((13/2)*len(models), 5), dpi = 300)
    fig.set_facecolor('white')

    for i, model in enumerate(models):

        model.eval()

        z = model.only_encoder(data).detach().numpy()
        z_label = labels.detach().numpy()

        if norm:
            z = (z - z.mean(axis=0)) / z.std(axis=0)

        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
        embedding = reducer.fit_transform(z)

        colors = ['red', 'blue', 'green', 'purple', 'orange']

        unique_labels = np.unique(z_label)

        for j, cls in enumerate(unique_labels):
            ax[i].scatter(embedding[z_label == cls, 0], embedding[z_label == cls, 1], c=[colors[j]], label=f'Class {cls}', s=10)


        ax[i].set_title(f'UMAP projection of the latent space of model {model.name}')
        ax[i].legend()
    
    plt.show()
    return fig

def plot_umap_lp(models, data, n_neighbors, min_dist, metric, norm = True):

    fig, ax = plt.subplots(1, len(models), figsize=((13/2)*len(models), 5), dpi = 300)
    fig.set_facecolor('white')

    for i, model in enumerate(models):

        model.eval()

        z_predicted = model(data[i].tensors[0]).detach().numpy()
        z_label = data[i].tensors[1].detach().numpy()

        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
        embedding = reducer.fit_transform(z_predicted)

        colors = ['red', 'blue', 'green', 'purple', 'orange']

        unique_labels = np.unique(z_label)

        for j, cls in enumerate(unique_labels):
            ax[i].scatter(embedding[z_label == cls, 0], embedding[z_label == cls, 1], c=[colors[j]], label=f'Class {cls}', s=10)


        ax[i].set_title(f'UMAP projection of the latent space of model') #{model.name}')
        ax[i].legend()
    
    plt.show()
    return fig

import seaborn as sns

def plot_matrix(models, matrix):

    fig, ax = plt.subplots(1, len(models), figsize=((13/2)*len(models), 5), dpi = 300)

    for i, model in enumerate(models):
        sns.heatmap(matrix[i], annot=True, ax=ax[i], fmt='d', cmap='Blues', cbar=False)
        ax[i].set_xlabel('Predicted')
        ax[i].set_ylabel('Real')
        ax[i].set_title(f'Confusion matrix of linear probing for model {model.name}')
    
    return fig