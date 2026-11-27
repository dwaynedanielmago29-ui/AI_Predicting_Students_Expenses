import numpy as np
from modules.modules_prediction import predict_new_sample

def test_prediction():
    # Sample beta coefficients for quick test
    test_beta = np.array([5.0, 0.4, 20.0, 12.0, 3.0])
    result = predict_new_sample(test_beta, 200, 2, 1, 7.5)
    print("Test Predicted Expense:", round(result, 2))
    assert result > 0

if __name__ == "__main__":
    test_prediction()
    