import numpy as np


def fit_linear_regression(X, y):
    """Fit OLS Linear Regression with an intercept using a stable least-squares solve."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X must be 2D and have the same number of rows as y.")
    X_b = np.column_stack((np.ones(X.shape[0]), X))
    beta, *_ = np.linalg.lstsq(X_b, y, rcond=None)
    return beta


def predict_regression(beta, X):
    """Predict target values from a fitted coefficient vector."""
    X = np.asarray(X, dtype=float)
    X_b = np.column_stack((np.ones(X.shape[0]), X))
    return X_b @ np.asarray(beta, dtype=float)
