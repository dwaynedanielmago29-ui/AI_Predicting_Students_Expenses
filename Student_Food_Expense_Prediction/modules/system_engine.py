"""Central integration layer for the Student Food Expense Prediction System."""
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from modules import dataset_manager as dm
from modules.modules_model_evaluation import calculate_rmse, calculate_r2

class FoodExpenseEngine:
    def __init__(self):
        self.df = dm.load_model_ready()
        self.validation = dm.validate_dataset(self.df)
        if not self.validation["is_valid"]:
            raise ValueError("Dataset validation failed: " + str(self.validation))
        self.prepared = dm.prepare_data(self.df)
        self.full_model = None
        self.pca_model = None
        self.scaler = self.prepared["scaler"]
        self.pca = None
        self.metrics = {}
        self._fit_models()

    def _fit_models(self):
        p = self.prepared
        self.full_model = LinearRegression().fit(p["X_train"], p["y_train"])
        full_pred = self.full_model.predict(p["X_test"])
        self.metrics["Full-feature Linear Regression"] = {
            "RMSE": calculate_rmse(p["y_test"], full_pred),
            "R2": calculate_r2(p["y_test"], full_pred),
        }
        self.pca = PCA(n_components=0.95)
        z_train = self.pca.fit_transform(p["X_train_std"])
        z_test = self.pca.transform(p["X_test_std"])
        self.pca_model = LinearRegression().fit(z_train, p["y_train"])
        pca_pred = self.pca_model.predict(z_test)
        self.metrics["PCA + Linear Regression"] = {
            "RMSE": calculate_rmse(p["y_test"], pca_pred),
            "R2": calculate_r2(p["y_test"], pca_pred),
        }
        self.pca_train = z_train
        self.pca_test = z_test
        self.full_predictions = full_pred
        self.pca_predictions = pca_pred

    def predict(self, allowance, meals, snacks, campus_hours, model="Full-feature Linear Regression"):
        values = np.array([[allowance, meals, snacks, campus_hours]], dtype=float)
        if model == "PCA + Linear Regression":
            transformed = self.pca.transform(self.scaler.transform(values))
            result = self.pca_model.predict(transformed)[0]
        else:
            result = self.full_model.predict(values)[0]
        return max(0.0, float(result))

    def matrix_report(self):
        x = self.prepared["X_train"]
        z = self.prepared["X_train_std"]
        return {
            "Training samples": x.shape[0],
            "Testing samples": self.prepared["X_test"].shape[0],
            "Feature count": x.shape[1],
            "Matrix rank": int(np.linalg.matrix_rank(x)),
            "Gram matrix": dm.gram_matrix(z),
            "Condition number (standardized X)": float(np.linalg.cond(z)),
            "Determinant sign (Gram)": int(np.linalg.slogdet(dm.gram_matrix(z))[0]),
        }

    def pca_summary(self):
        return self.pca.explained_variance_ratio_, self.pca.explained_variance_ratio_.cumsum()

    def export_predictions(self, path):
        import pandas as pd
        out = pd.DataFrame({
            "RespondentID": self.prepared["test_df"]["RespondentID"].values,
            "Actual_DailyFoodExpense_PHP": self.prepared["y_test"].values,
            "Predicted_FullFeature_PHP": self.full_predictions,
            "Predicted_PCA_PHP": self.pca_predictions,
        })
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
        return Path(path)
