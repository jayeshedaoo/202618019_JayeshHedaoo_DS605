import numpy as np


def manual_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def manual_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def manual_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - (ss_res / ss_tot)


def confusion_values(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return tp, tn, fp, fn


def manual_accuracy(y_true, y_pred):
    tp, tn, fp, fn = confusion_values(y_true, y_pred)

    return (tp + tn) / (tp + tn + fp + fn)


def manual_precision(y_true, y_pred):
    tp, tn, fp, fn = confusion_values(y_true, y_pred)

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def manual_recall(y_true, y_pred):
    tp, tn, fp, fn = confusion_values(y_true, y_pred)

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def manual_f1(y_true, y_pred):
    precision = manual_precision(y_true, y_pred)
    recall = manual_recall(y_true, y_pred)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def train_logistic_regression(
    X,
    y,
    learning_rate=0.01,
    epochs=2000
):
    weights = np.zeros(X.shape[1])

    n_samples = X.shape[0]

    for _ in range(epochs):
        z = X @ weights
        probabilities = sigmoid(z)

        gradient = (
            X.T @ (probabilities - y)
        ) / n_samples

        weights -= learning_rate * gradient

    return weights