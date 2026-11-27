"""
Main application for Student Daily Food Expense Prediction.

This file integrates the existing dataset, regression, evaluation, and
prediction modules into a Tkinter desktop application.

Expected project layout:
Student_Food_Expense_Prediction/
    main.py
    dataset_manager.py
    data/
        Student_Daily_Food_Expenses_Model_Ready.csv
    modules/
        __init__.py
        modules_regression.py
        modules_model_evaluation.py
        modules_prediction.py
"""

from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd

from dataset_manager import (
    load_model_ready,
    validate_dataset,
    prepare_data,
    FEATURES,
    TARGET,
)

from modules.modules_regression import (
    fit_linear_regression,
    predict_regression,
)
from modules.modules_model_evaluation import (
    calculate_rmse,
    calculate_r2,
)
from modules.modules_prediction import predict_new_sample


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "Student_Daily_Food_Expenses_Model_Ready.csv"
OUTPUT_DIR = BASE_DIR / "outputs"


def calculate_safe_r2(y_true, y_pred):
    """
    Call the group's evaluation function. The current shared version of
    calculate_r2 may accidentally return the function instead of the score.
    This fallback keeps the application usable until that module is fixed.
    """
    result = calculate_r2(y_true, y_pred)
    if isinstance(result, (int, float, np.integer, np.floating)):
        return float(result)

    # Correct R-squared calculation as a temporary integration fallback.
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - (ss_res / ss_tot))


class FoodExpenseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Daily Food Expense Prediction")
        self.root.geometry("1000x700")
        self.root.minsize(850, 600)

        self.dataset = None
        self.prepared = None
        self.full_beta = None
        self.pca_beta = None
        self.pca_components = None
        self.pca_k = None
        self.scaler = None
        self.last_results = None

        self._build_interface()

    def _build_interface(self):
        header = ttk.Frame(self.root, padding=16)
        header.pack(fill="x")

        ttk.Label(
            header,
            text="Student Daily Food Expense Prediction",
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            header,
            text="Linear Algebra • PCA • Multiple Linear Regression",
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(4, 0))

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        self.dashboard_tab = ttk.Frame(self.notebook, padding=14)
        self.analysis_tab = ttk.Frame(self.notebook, padding=14)
        self.prediction_tab = ttk.Frame(self.notebook, padding=14)
        self.notebook.add(self.dashboard_tab, text="Dashboard")
        self.notebook.add(self.analysis_tab, text="Model Analysis")
        self.notebook.add(self.prediction_tab, text="Prediction")

        self._build_dashboard()
        self._build_analysis()
        self._build_prediction()

    def _build_dashboard(self):
        ttk.Label(
            self.dashboard_tab,
            text="Project Overview",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w")

        description = (
            "This application estimates a student's daily food expenses "
            "using daily allowance and spending habits. It compares a "
            "full-feature regression model with a PCA-reduced regression model."
        )
        ttk.Label(
            self.dashboard_tab,
            text=description,
            wraplength=850,
            justify="left",
        ).pack(anchor="w", pady=(8, 18))

        info = ttk.LabelFrame(self.dashboard_tab, text="Input Features", padding=12)
        info.pack(fill="x", pady=6)
        for item in [
            "Daily allowance (PHP)",
            "Number of meals purchased",
            "Number of snacks purchased",
            "Hours spent on campus",
        ]:
            ttk.Label(info, text=f"• {item}").pack(anchor="w", pady=2)

        self.dataset_status = tk.StringVar(value="Dataset not loaded yet.")
        ttk.Label(
            self.dashboard_tab,
            textvariable=self.dataset_status,
            foreground="#245b8f",
            wraplength=850,
        ).pack(anchor="w", pady=(16, 8))

        buttons = ttk.Frame(self.dashboard_tab)
        buttons.pack(anchor="w", pady=8)
        ttk.Button(
            buttons,
            text="Load Dataset and Run Models",
            command=self.run_analysis,
        ).pack(side="left", padx=(0, 8))
        ttk.Button(
            buttons,
            text="Validate Dataset",
            command=self.validate_only,
        ).pack(side="left")

    def _build_analysis(self):
        ttk.Label(
            self.analysis_tab,
            text="Model Evaluation",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            self.analysis_tab,
            text="The test set is kept separate from training. Standardization "
                 "parameters and PCA components are learned from training data only.",
            wraplength=850,
        ).pack(anchor="w", pady=(6, 12))

        self.results_text = tk.Text(
            self.analysis_tab,
            height=25,
            wrap="word",
            font=("Consolas", 10),
        )
        self.results_text.pack(fill="both", expand=True)
        self.results_text.insert(
            "1.0",
            "Click 'Load Dataset and Run Models' on the Dashboard tab.\n",
        )
        self.results_text.configure(state="disabled")

    def _build_prediction(self):
        ttk.Label(
            self.prediction_tab,
            text="Predict Daily Food Expense",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            self.prediction_tab,
            text="Enter the student's information. Use the same units as the training dataset.",
            wraplength=850,
        ).pack(anchor="w", pady=(6, 14))

        form = ttk.Frame(self.prediction_tab)
        form.pack(anchor="w", fill="x")

        self.allowance_var = tk.StringVar()
        self.meals_var = tk.StringVar()
        self.snacks_var = tk.StringVar()
        self.hours_var = tk.StringVar()
        self.model_var = tk.StringVar(value="Full-feature model")

        fields = [
            ("Daily allowance (PHP)", self.allowance_var),
            ("Meals purchased", self.meals_var),
            ("Snacks purchased", self.snacks_var),
            ("Hours on campus", self.hours_var),
        ]
        for row, (label, variable) in enumerate(fields):
            ttk.Label(form, text=label, width=24).grid(
                row=row, column=0, sticky="w", padx=(0, 10), pady=7
            )
            ttk.Entry(form, textvariable=variable, width=24).grid(
                row=row, column=1, sticky="w", pady=7
            )

        ttk.Label(form, text="Model", width=24).grid(
            row=4, column=0, sticky="w", padx=(0, 10), pady=7
        )
        model_select = ttk.Combobox(
            form,
            textvariable=self.model_var,
            values=["Full-feature model", "PCA-reduced model"],
            state="readonly",
            width=22,
        )
        model_select.grid(row=4, column=1, sticky="w", pady=7)

        ttk.Button(
            form,
            text="Predict Expense",
            command=self.predict_expense,
        ).grid(row=5, column=1, sticky="w", pady=(14, 6))

        self.prediction_result = tk.StringVar(value="Prediction will appear here.")
        ttk.Label(
            self.prediction_tab,
            textvariable=self.prediction_result,
            font=("Segoe UI", 13, "bold"),
            foreground="#176b45",
            wraplength=850,
        ).pack(anchor="w", pady=18)

        ttk.Label(
            self.prediction_tab,
            text="Note: Predictions are estimates based on the collected survey data; "
                 "they are not guaranteed actual expenses.",
            wraplength=850,
        ).pack(anchor="w")

    def _set_results(self, text):
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", text)
        self.results_text.configure(state="disabled")

    def validate_only(self):
        try:
            df = load_model_ready(DATASET_PATH)
            report = validate_dataset(df)
            lines = [
                "DATASET VALIDATION",
                "=" * 55,
                f"Rows: {report['rows']}",
                f"Columns: {report['columns']}",
                f"Missing columns: {report['missing_columns']}",
                f"Non-numeric columns: {report['non_numeric_columns']}",
                f"Negative values: {report['negative_values']}",
                f"Duplicate IDs: {report['duplicate_ids']}",
                f"Duplicate rows: {report['duplicate_rows']}",
                f"Zero-variance columns: {report['zero_variance_columns']}",
                f"Valid: {report['is_valid']}",
            ]
            self._set_results("\n".join(lines))
            self.notebook.select(self.analysis_tab)
            self.dataset_status.set(
                f"Dataset found: {len(df)} rows. Validation: "
                f"{'PASSED' if report['is_valid'] else 'CHECK REQUIRED'}."
            )
        except Exception as exc:
            messagebox.showerror("Dataset Error", str(exc))

    def run_analysis(self):
        try:
            if not DATASET_PATH.exists():
                raise FileNotFoundError(
                    "Dataset CSV was not found.\n\nExpected file:\n"
                    f"{DATASET_PATH}\n\n"
                    "Place Student_Daily_Food_Expenses_Model_Ready.csv "
                    "inside the data folder."
                )

            df = load_model_ready(DATASET_PATH)
            report = validate_dataset(df)
            if not report["is_valid"]:
                raise ValueError(
                    "The dataset did not pass validation. Check the CSV "
                    "columns, missing values, negative values, and duplicates."
                )

            prepared = prepare_data(df)
            X_train_std = prepared["X_train_std"]
            X_test_std = prepared["X_test_std"]
            y_train = prepared["y_train"]
            y_test = prepared["y_test"]

            # Full-feature regression model (all four standardized features).
            full_beta = fit_linear_regression(X_train_std, y_train)
            full_pred = predict_regression(full_beta, X_test_std)
            full_rmse = float(calculate_rmse(y_test, full_pred))
            full_r2 = calculate_safe_r2(y_test, full_pred)

            # PCA is fitted using standardized TRAINING data only.
            covariance = np.cov(X_train_std, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

            total = np.sum(eigenvalues)
            if total <= 0:
                raise ValueError("PCA could not be computed: total variance is zero.")

            explained = eigenvalues / total
            cumulative = np.cumsum(explained)
            k = int(np.searchsorted(cumulative, 0.95) + 1)
            k = max(1, min(k, X_train_std.shape[1]))

            components = eigenvectors[:, :k]
            X_train_pca = X_train_std @ components
            X_test_pca = X_test_std @ components

            pca_beta = fit_linear_regression(X_train_pca, y_train)
            pca_pred = predict_regression(pca_beta, X_test_pca)
            pca_rmse = float(calculate_rmse(y_test, pca_pred))
            pca_r2 = calculate_safe_r2(y_test, pca_pred)

            self.dataset = df
            self.prepared = prepared
            self.full_beta = full_beta
            self.pca_beta = pca_beta
            self.pca_components = components
            self.pca_k = k
            self.scaler = prepared["scaler"]
            self.last_results = {
                "full_rmse": full_rmse,
                "full_r2": full_r2,
                "pca_rmse": pca_rmse,
                "pca_r2": pca_r2,
                "k": k,
                "explained": explained,
            }

            lines = [
                "MODEL EVALUATION RESULTS",
                "=" * 65,
                f"Dataset rows: {len(df)}",
                f"Training samples: {len(y_train)}",
                f"Testing samples: {len(y_test)}",
                f"Number of original features (p): {X_train_std.shape[1]}",
                "",
                "FULL STANDARDIZED REGRESSION MODEL",
                f"Test RMSE: PHP {full_rmse:,.2f}",
                f"Test R-squared: {full_r2:.4f}",
                f"Beta coefficients (intercept first): {np.round(full_beta, 4)}",
                "",
                "PCA-REDUCED REGRESSION MODEL",
                f"Number of retained components (k): {k}",
                f"Explained variance by component: {np.round(explained, 4)}",
                f"Test RMSE: PHP {pca_rmse:,.2f}",
                f"Test R-squared: {pca_r2:.4f}",
                "",
                "COMPARISON",
                "-" * 65,
                f"{'Model':30s} {'RMSE (PHP)':>14s} {'R-squared':>12s}",
                f"{'Full standardized':30s} {full_rmse:14.2f} {full_r2:12.4f}",
                f"{'PCA-reduced':30s} {pca_rmse:14.2f} {pca_r2:12.4f}",
                "",
                "Interpretation: Lower RMSE indicates smaller prediction errors. "
                "Higher R-squared indicates that the model explains more variation "
                "in the test-set target values.",
                "",
                "PCA components and standardization values were learned from "
                "training data only.",
            ]

            self._set_results("\n".join(lines))
            self.dataset_status.set(
                f"Analysis completed using {len(df)} records "
                f"({len(y_train)} training / {len(y_test)} testing)."
            )
            self.notebook.select(self.analysis_tab)
            messagebox.showinfo("Analysis Complete", "Both regression models were evaluated.")
        except Exception as exc:
            messagebox.showerror("Analysis Error", str(exc))

    def predict_expense(self):
        if self.full_beta is None or self.pca_beta is None:
            messagebox.showwarning(
                "Models Not Ready",
                "Please load the dataset and run model analysis first.",
            )
            return

        try:
            allowance = float(self.allowance_var.get())
            meals = float(self.meals_var.get())
            snacks = float(self.snacks_var.get())
            campus_hours = float(self.hours_var.get())

            values = np.array([[allowance, meals, snacks, campus_hours]], dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError("Please enter valid numeric values.")
            if np.any(values < 0):
                raise ValueError("Inputs cannot be negative.")

            # Apply the exact training-set standardization to the new sample.
            standardized = self.scaler.transform(values)

            if self.model_var.get() == "Full-feature model":
                # Convert standardized-model coefficients back to raw-input
                # coefficients so the shared prediction module can receive
                # allowance, meals, snacks, and campus hours in their usual units.
                raw_beta = np.empty_like(self.full_beta, dtype=float)
                raw_beta[1:] = self.full_beta[1:] / self.scaler.sigma
                raw_beta[0] = self.full_beta[0] - np.sum(
                    self.full_beta[1:] * self.scaler.mu / self.scaler.sigma
                )
                prediction = predict_new_sample(
                    raw_beta, allowance, meals, snacks, campus_hours
                )
            else:
                pca_sample = standardized @ self.pca_components
                prediction = float(
                    predict_regression(self.pca_beta, pca_sample)[0]
                )

            self.prediction_result.set(
                f"Estimated daily food expense: PHP {max(0.0, prediction):,.2f}"
            )
        except ValueError as exc:
            messagebox.showerror("Invalid Input", str(exc))
        except Exception as exc:
            messagebox.showerror("Prediction Error", str(exc))


def main():
    root = tk.Tk()
    app = FoodExpenseApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
