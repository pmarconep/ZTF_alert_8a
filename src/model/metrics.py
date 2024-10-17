import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def performance_metrics(TP, FP, FN, TN):
    """Calcula metricas de desempeño.

    Args:
        TP: Numero de verdaderos positivos.
        FP: Numero de falsos positivos.
        FN: Numero de falsos negativos.
        TN: Numero de verdaderos negativos.

    Returns:
        accuracy: Porcentaje de clasificaciones correctas del detector.
        precision: Precision del detector.
        recall: Recall/Sensibilidad del detector.
    """
    accuracy = 100.0 * (TP + TN) / (TP + TN + FP + FN)
    precision = 100.0 * TP / (TP + FP)
    recall = 100.0 * TP / (TP + FN)
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    print(f"{accuracy:.4f} Accuracy (Porcentaje de clasificaciones correctas)")
    print(f"{precision:.4f} Precision")
    print(f"{recall:.4f} Recall")
    print()
    return accuracy, precision, recall

def roc_curve(labels, probabilities):
    """Calcula la curva ROC.

    Args:
        labels: Array binario 1-D con las etiquetas reales.
        probabilities: Array 1-D continuo en el rango [0, 1] con las
            probabilidades de la clase 1.

    Returns:
        tpr: Array 1-D con los valores de Tasa de Verdaderos Positivos (TPR).
        fpr: Array 1-D con los valores de Tasa de Falsos Positivos (FPR).
    """
    tpr = []
    fpr = []
    for threshold in np.linspace(0, 1, 1000):
        TN, FP, FN, TP = conf_matrix_given_threshold(labels, probabilities, threshold)
        tpr.append(TP / (TP + FN))
        fpr.append(FP / (FP + TN))

    return np.array(tpr), np.array(fpr)

def conf_matrix_given_threshold(true_labels, prediction, threshold):
    probabilities_with_threshold = (prediction > threshold).long()
    TN, FP, FN, TP = confusion_matrix(true_labels, probabilities_with_threshold).ravel()
    return TP, FP, FN, TN

def show_curves(curves, models):
    fig, ax = plt.subplots(1, len(curves), figsize=((13/2)*len(curves), 5), dpi = 300)
    fig.set_facecolor('white')

    epochs = [np.arange(len(curve["val_loss"])) + 1 for curve in curves]

    for i, curve in enumerate(curves):
        ax[i].plot(epochs[i], curve['val_loss'], label='validation')
        ax[i].plot(epochs[i], curve['train_loss'], label='training')
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