import numpy as np

def fit_linear_regression(X, y):
    """
    Fit a multiple linear regression model.
    X: predictor matrix (allowance, meals, snacks, campus hours)
    y: target vector (daily food expense)
    Returns: beta coefficients including intercept
    """
    # Add column of ones for intercept
    ones = np.ones((X.shape[0], 1))
    X_b = np.hstack((ones, X))

    # Formula: beta = (X^T X)^(-1) X^T y
    Xt = X_b.T
    beta = np.linalg.inv(Xt @ X_b) @ Xt @ y
    return beta


def predict_regression(beta, X):
    """
    Predict food expense using the fitted beta coefficients.
    """
    ones = np.ones((X.shape[0], 1))
    X_b = np.hstack((ones, X))
    y_pred = X_b @ beta
    return y_pred
    