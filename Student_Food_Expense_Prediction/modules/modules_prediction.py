import numpy as np

def predict_new_sample(beta, allowance, meals, snacks, campus_hours):
    """
    Predict the daily food expense for one new student sample.
    beta: the fitted regression coefficients
    """
    x = np.array([[1, allowance, meals, snacks, campus_hours]])
    predicted_expense = x @ beta
    return float(predicted_expense[0])
