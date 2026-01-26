import numpy as np


def fit_scaler(X):
    mean = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1
    return mean, stds


def transform(X, mean, stds):
    return (X - mean) / stds


def fit_transform(X):
    means, stds = fit_scaler(X)
    X_scaled = transform(X, means, stds)
    return X_scaled, means, stds
