import numpy as np


def train_test_split(X, y, test_size=0.2, shuffle=True):
    n = len(X)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)

    split_idx = int(n * (1 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
