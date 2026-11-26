import numpy as np

def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error
    """
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def calculate_r2(y_true, y_pred):
    """
    Calculate R-squared score
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return calculate_r2