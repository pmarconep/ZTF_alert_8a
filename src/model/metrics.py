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
    return TN, FP, FN, TP

def show_curves(curves):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.set_facecolor('white')

    epochs = np.arange(len(curves["val_loss"])) + 1

    ax[0].plot(epochs, curves['val_loss'], label='validation')
    ax[0].plot(epochs, curves['train_loss'], label='training')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss evolution during training')
    ax[0].legend()

    # ax[1].plot(epochs, curves['val_acc'], label='validation')
    # ax[1].plot(epochs, curves['train_acc'], label='training')
    # ax[1].set_xlabel('Epoch')
    # ax[1].set_ylabel('Accuracy')
    # ax[1].set_title('Accuracy evolution during training')
    # ax[1].legend()

    plt.show()
    return fig

def show_loss(curves1, curves2, curves3, curves4, curves5, curvesGauss):
    fig, ax = plt.subplots(1, 6, figsize=(32.5, 5))
    fig.set_facecolor('white')

    epochs = np.arange(len(curves1["val_loss"])) + 1

    curves = [curves1, curves2, curves3, curves4, curves5, curvesGauss]
    
    for i in range(6):
        ax[i].plot(epochs, curves[i]['val_loss'], label='validation')
        ax[i].plot(epochs, curves[i]['train_loss'], label='training')
        ax[i].set_xlabel('Epoch')
        ax[i].set_ylabel('Loss')
        ax[i].set_title(f'Loss evolution during training (Model {i})')
        ax[i].legend()

    plt.show()
    return fig

import umap

def plot_umap(model, data, labels, n_neighbors, min_dist, metric, norm = True):
    model.eval()

    z = model.only_encoder(data).detach().numpy()
    z_label = labels.detach().numpy()

    if norm:
        z = (z - z.mean(axis=0)) / z.std(axis=0)

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    embedding = reducer.fit_transform(z)

    colors = ['red', 'blue', 'green', 'purple', 'orange']

    fig = plt.figure(figsize=(10, 8))
    for i, color in enumerate(colors):
        plt.scatter(embedding[z_label == i, 0], embedding[z_label == i, 1], c=color, label=f'Class {i}', s=5)
    # plt.title(f'UMAP projection of the latent space of model {model.name}')
    plt.legend()
    plt.show()
    return fig