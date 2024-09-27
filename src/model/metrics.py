import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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