"""Student Food Expense Prediction System - integrated Tkinter application.

Connects the existing project modules to the original simple interface.
The workflow follows the manuscript: 80:20 split (seed 42), train-only
standardization, matrix diagnostics, PCA >=95% variance, two OLS models,
RMSE/R2 comparison, and new-student inference.
"""
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import pandas as pd

from modules import dataset_manager as dm
from modules import data_exploration as de
from modules import matrix_analysis as ma
from modules import determinant_analysis as da
from modules import vector_space as vs
from modules import transformations as tf
from modules import preprocessing as prep
from modules.eigen_pca import EigenPCA
from modules import modules_regression as regression
from modules import modules_model_evaluation as evaluation
from modules import modules_prediction as prediction
from modules import modules_report_export as report_export
from modules.visualization import Visualizer

APP_TITLE = "Student Food Expense Prediction System"
BG = "#F3F6FC"
NAVY = "#17243A"
NAVY_LIGHT = "#263954"
BLUE = "#4169E1"
TEAL = "#20B8A5"
TEXT = "#1E293B"
MUTED = "#718096"
WHITE = "#FFFFFF"
BORDER = "#E2E8F0"

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


def load_project_dataset():
    """Load expected CSV, with compatibility for the supplied '(1)' filename."""
    candidates = [
        BASE_DIR / "data" / "Student_Daily_Food_Expenses_Model_Ready.csv",
        BASE_DIR / "data" / "Student_Daily_Food_Expenses_Model_Ready (1).csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            # Keep only the expected research columns, preserving respondent ID.
            required = [dm.ID_COLUMN] + dm.NUMERIC_COLUMNS
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Dataset is missing required columns: {missing}")
            return df[required].copy(), path
    raise FileNotFoundError(
        "Dataset CSV not found. Place Student_Daily_Food_Expenses_Model_Ready.csv "
        "inside the data folder."
    )


def fit_ols(X, y):
    """Use the project's regression module with a stable least-squares solve."""
    return regression.fit_linear_regression(np.asarray(X, float), np.asarray(y, float))


class FoodExpenseApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1160x720")
        self.minsize(960, 600)
        self.configure(bg=BG)
        self.pages = [
            ("Dashboard", "⌂"), ("Dataset Management", "▦"),
            ("Data Exploration", "◫"), ("Linear Algebra", "∑"),
            ("Eigenvalues & PCA", "λ"), ("Regression Analysis", "ƒ"),
            ("Prediction", "↗"),
        ]
        self.buttons = {}
        self.df = None
        self.dataset_path = None
        self.analysis = None
        self.models = None
        self.prediction_history = []
        self._load_and_train()
        self._build_shell()
        self.show_page("Dashboard")

    def _load_and_train(self):
        self.df, self.dataset_path = load_project_dataset()
        # Connect the shared preprocessing utility and verify feature/target separation.
        X_check, y_check = prep.split_features_target(self.df[dm.FEATURES + [dm.TARGET]], dm.TARGET)
        if list(X_check.columns) != dm.FEATURES or len(y_check) != len(self.df):
            raise ValueError("Preprocessing feature/target separation did not match the manuscript schema.")
        validation = dm.validate_dataset(self.df)
        if not validation["is_valid"]:
            problems = []
            if validation["missing_columns"]: problems.append(f"Missing columns: {validation['missing_columns']}")
            if validation["missing_percent"]: 
                missing = {k: v for k, v in validation["missing_percent"].items() if v > 0}
                if missing: problems.append(f"Missing values: {missing}")
            if validation["negative_values"]: problems.append(f"Negative values: {validation['negative_values']}")
            if validation["zero_variance_columns"]: problems.append(f"Constant columns: {validation['zero_variance_columns']}")
            if validation["duplicate_ids"]: problems.append(f"Duplicate IDs: {validation['duplicate_ids']}")
            raise ValueError("Dataset validation failed. " + "; ".join(problems))

        # Manuscript Section 2.2: same 80:20 split and seed 42 for both models.
        data = dm.prepare_data(self.df, test_size=0.20, seed=42)
        diag = de.matrix_diagnostics(data["X_train"], data["X_train_std"])
        eig = de.eigen_analysis(data["X_train_std"], variance_target=0.95)
        Z_train, Z_test = de.project_pca(
            data["X_train_std"], data["X_test_std"],
            eig["eigenvectors"], eig["k_selected"]
        )

        # Fit PCA class as a second module-level implementation/check.
        pca_check = EigenPCA(n_components=0.95).fit(data["X_train_std"])
        if pca_check.components.shape[1] != eig["k_selected"]:
            raise RuntimeError("PCA implementations selected different component counts.")

        beta_full = fit_ols(data["X_train_std"], data["y_train"])
        beta_pca = fit_ols(Z_train, data["y_train"])
        pred_full = regression.predict_regression(beta_full, data["X_test_std"])
        pred_pca = regression.predict_regression(beta_pca, Z_test)
        results = {
            "Full standardized": {
                "beta": beta_full, "predictions": pred_full,
                "rmse": float(evaluation.calculate_rmse(data["y_test"], pred_full)),
                "r2": float(evaluation.calculate_r2(data["y_test"], pred_full)),
            },
            "PCA-reduced": {
                "beta": beta_pca, "predictions": pred_pca,
                "rmse": float(evaluation.calculate_rmse(data["y_test"], pred_pca)),
                "r2": float(evaluation.calculate_r2(data["y_test"], pred_pca)),
            },
        }
        self.analysis = {"data": data, "diagnostics": diag, "eigen": eig,
                         "Z_train": Z_train, "Z_test": Z_test, "pca_check": pca_check}
        self.models = results

    def _build_shell(self):
        self.sidebar = tk.Frame(self, bg=NAVY, width=238)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        brand = tk.Frame(self.sidebar, bg=NAVY)
        brand.pack(fill="x", padx=20, pady=(25, 30))
        tk.Label(brand, text="∑", font=("Cambria Math", 28, "bold"), fg=WHITE,
                 bg=BLUE, width=2, pady=3).pack(side="left", padx=(0, 10))
        words = tk.Frame(brand, bg=NAVY); words.pack(side="left")
        tk.Label(words, text="FOOD EXPENSE", fg=WHITE, bg=NAVY,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tk.Label(words, text="PREDICTION SYSTEM", fg="#AAB8CC", bg=NAVY,
                 font=("Segoe UI", 8)).pack(anchor="w", pady=(3, 0))
        tk.Label(self.sidebar, text="WORKSPACE", fg="#8FA1BA", bg=NAVY,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=22, pady=(0, 10))
        for name, symbol in self.pages:
            btn = tk.Button(self.sidebar, text=f"  {symbol}     {name}", anchor="w",
                font=("Segoe UI", 10), fg="#DCE5F2", bg=NAVY,
                activeforeground=WHITE, activebackground=NAVY_LIGHT, relief="flat",
                bd=0, padx=15, pady=12, cursor="hand2",
                command=lambda page=name: self.show_page(page))
            btn.pack(fill="x", padx=11, pady=2); self.buttons[name] = btn
        tk.Frame(self.sidebar, bg="#34445D", height=1).pack(side="bottom", fill="x", padx=18, pady=(10, 12))
        tk.Label(self.sidebar, text="LINEAR ALGEBRA  •  PCA  •  REGRESSION",
                 fg="#C5D0DF", bg=NAVY, font=("Segoe UI", 7, "bold")).pack(side="bottom", anchor="w", padx=18, pady=(0, 18))
        self.main_area = tk.Frame(self, bg=BG); self.main_area.pack(side="right", fill="both", expand=True)
        top = tk.Frame(self.main_area, bg=WHITE, height=64, highlightbackground=BORDER, highlightthickness=1)
        top.pack(fill="x"); top.pack_propagate(False)
        self.heading = tk.Label(top, text="Dashboard", font=("Segoe UI", 16, "bold"), fg=TEXT, bg=WHITE)
        self.heading.pack(side="left", padx=28)
        tk.Label(top, text="STUDENT ANALYTICS", font=("Segoe UI", 8, "bold"), fg=BLUE,
                 bg="#EDF2FF", padx=12, pady=7).pack(side="right", padx=25)
        self.body = tk.Frame(self.main_area, bg=BG); self.body.pack(fill="both", expand=True, padx=28, pady=22)

    def _clear_body(self):
        for child in self.body.winfo_children(): child.destroy()

    def _heading(self, title, subtitle):
        tk.Label(self.body, text=title, font=("Segoe UI", 21, "bold"), fg=TEXT, bg=BG).pack(anchor="w", pady=(5, 4))
        tk.Label(self.body, text=subtitle, font=("Segoe UI", 10), fg=MUTED, bg=BG).pack(anchor="w", pady=(0, 16))

    def _card(self, parent, title, value, detail, accent):
        card = tk.Frame(parent, bg=WHITE, highlightbackground=BORDER, highlightthickness=1)
        tk.Frame(card, bg=accent, height=4).pack(fill="x")
        inner = tk.Frame(card, bg=WHITE); inner.pack(fill="both", expand=True, padx=15, pady=13)
        tk.Label(inner, text=title.upper(), font=("Segoe UI", 8, "bold"), fg=MUTED, bg=WHITE).pack(anchor="w")
        tk.Label(inner, text=value, font=("Segoe UI", 20, "bold"), fg=TEXT, bg=WHITE).pack(anchor="w", pady=(6, 2))
        tk.Label(inner, text=detail, font=("Segoe UI", 8), fg=MUTED, bg=WHITE).pack(anchor="w")
        return card

    def _text(self, parent, content, height=16):
        wrap = tk.Frame(parent, bg=WHITE, highlightbackground=BORDER, highlightthickness=1)
        txt = tk.Text(wrap, height=height, wrap="word", font=("Consolas", 9),
                      bg=WHITE, fg=TEXT, relief="flat", padx=12, pady=10)
        scroll = ttk.Scrollbar(wrap, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=scroll.set)
        txt.pack(side="left", fill="both", expand=True); scroll.pack(side="right", fill="y")
        txt.insert("1.0", content); txt.configure(state="disabled")
        return wrap

    def _tree(self, parent, columns, headings, rows, height=10):
        frame = tk.Frame(parent, bg=WHITE)
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=height)
        for col, title in zip(columns, headings):
            tree.heading(col, text=title); tree.column(col, width=max(110, int(800/len(columns))), anchor="center")
        for row in rows: tree.insert("", "end", values=row)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview); tree.configure(yscrollcommand=scroll.set)
        tree.pack(side="left", fill="both", expand=True); scroll.pack(side="right", fill="y")
        return frame

    def show_page(self, page):
        self.heading.config(text=page)
        for name, button in self.buttons.items():
            button.config(bg=BLUE if name == page else NAVY, fg=WHITE if name == page else "#DCE5F2")
        self._clear_body()
        actions = {"Dashboard": self._dashboard, "Dataset Management": self._dataset_page,
                   "Data Exploration": self._exploration_page, "Linear Algebra": self._linear_algebra_page,
                   "Eigenvalues & PCA": self._pca_page, "Regression Analysis": self._regression_page,
                   "Prediction": self._prediction_page}
        actions[page]()

    def _dashboard(self):
        self._heading("Welcome to your analysis workspace", "Student food expense analysis using the collected survey dataset.")
        d, e = self.analysis["data"], self.analysis["eigen"]
        cards = tk.Frame(self.body, bg=BG); cards.pack(fill="x")
        items = [("COMPLETE RESPONSES", str(len(self.df)), "Research dataset", BLUE),
                 ("TRAIN / TEST", f"{len(d['train_df'])} / {len(d['test_df'])}", "80:20 split · seed 42", TEAL),
                 ("PCA COMPONENTS", str(e["k_selected"]), f"{e['cumulative_ratio'][e['k_selected']-1]*100:.2f}% variance", "#8B5CF6")]
        for i, item in enumerate(items):
            self._card(cards, *item).grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 10, 0))
            cards.grid_columnconfigure(i, weight=1)
        lower = tk.Frame(self.body, bg=BG); lower.pack(fill="both", expand=True, pady=(18, 0))
        overview = tk.Frame(lower, bg=WHITE, highlightbackground=BORDER, highlightthickness=1)
        overview.pack(side="left", fill="both", expand=True, padx=(0, 10))
        tk.Label(overview, text="PROJECT OVERVIEW", font=("Segoe UI", 9, "bold"), fg=BLUE, bg=WHITE).pack(anchor="w", padx=20, pady=(18, 10))
        tk.Label(overview, text=("This system uses daily allowance, meals purchased, snacks purchased, and campus hours "
            "to estimate a student's daily food expense. It applies matrix analysis, eigen-analysis, PCA, and two "
            "Linear Regression models. The test set is kept separate from training."),
            font=("Segoe UI", 10), fg=TEXT, bg=WHITE, wraplength=500, justify="left").pack(anchor="w", padx=20, pady=(0, 15))
        quick = tk.Frame(lower, bg=WHITE, width=245, highlightbackground=BORDER, highlightthickness=1)
        quick.pack(side="right", fill="y"); quick.pack_propagate(False)
        tk.Label(quick, text="QUICK NAVIGATION", font=("Segoe UI", 9, "bold"), fg=BLUE, bg=WHITE).pack(anchor="w", padx=17, pady=(18, 12))
        for label, target in [("Open Dataset", "Dataset Management"), ("View Matrix Analysis", "Linear Algebra"), ("View PCA Results", "Eigenvalues & PCA"), ("Compare Models", "Regression Analysis"), ("Make Prediction", "Prediction")]:
            tk.Button(quick, text=label, anchor="w", font=("Segoe UI", 9), fg=TEXT, bg="#F4F7FC", relief="flat", padx=10, pady=9,
                      command=lambda p=target: self.show_page(p)).pack(fill="x", padx=12, pady=3)

    def _dataset_page(self):
        self._heading("Dataset Management", f"Loaded from: {self.dataset_path.relative_to(BASE_DIR)}")
        report = dm.validate_dataset(self.df)
        summary = (f"Rows: {report['rows']}\nColumns: {report['columns']}\n"
                   f"Required rows expected by manuscript: 98\nDataset validation: {'VALID' if report['is_valid'] else 'CHECK REQUIRED'}\n"
                   f"Missing values: {int(self.df[dm.NUMERIC_COLUMNS].isna().sum().sum())}\n"
                   f"Duplicate rows: {report['duplicate_rows']}\nDuplicate IDs: {report['duplicate_ids']}\n")
        self._text(self.body, summary, 7).pack(fill="x", pady=(0, 12))
        rows = [tuple(row) for row in self.df.head(12)[[dm.ID_COLUMN] + dm.NUMERIC_COLUMNS].itertuples(index=False, name=None)]
        self._tree(self.body, ["id"] + dm.NUMERIC_COLUMNS, ["RESPONDENT"] + dm.NUMERIC_COLUMNS, rows, 9).pack(fill="both", expand=True)
        tk.Button(self.body, text="Reload Dataset and Recalculate Models", command=self._reload,
                  bg=BLUE, fg=WHITE, relief="flat", padx=12, pady=8).pack(anchor="w", pady=10)

    def _reload(self):
        try:
            self._load_and_train(); self.show_page("Dataset Management")
            messagebox.showinfo("Updated", "Dataset reloaded and both models recalculated.")
        except Exception as exc: messagebox.showerror("Reload failed", str(exc))

    def _exploration_page(self):
        self._heading("Data Exploration", "Descriptive statistics, missing values, correlations, and outlier screening.")
        stats = de.summary_statistics(self.df)
        tk.Label(self.body, text="Summary statistics", bg=BG, fg=TEXT, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        rows = [(idx, *[stats.loc[idx, c] for c in stats.columns]) for idx in stats.index]
        self._tree(self.body, ["feature", "count", "mean", "std", "min", "median", "max"],
                   ["VARIABLE", "N", "MEAN", "SD", "MIN", "MEDIAN", "MAX"], rows, 6).pack(fill="x", pady=(5, 12))
        corr = de.correlation_matrix(self.df)
        corr_text = "CORRELATION MATRIX\n" + corr.to_string() + "\n\nCORRELATION WITH TARGET\n" + de.target_correlations(self.df).to_string()
        self._text(self.body, corr_text, 11).pack(fill="both", expand=True)

    def _linear_algebra_page(self):
        self._heading("Linear Algebra", "Matrix representation, vector-space properties, determinants, and numerical conditioning.")
        d, diag = self.analysis["data"], self.analysis["diagnostics"]
        Xtr, Ztr = d["X_train"], d["X_train_std"]
        gram = ma.matrix_multiply(ma.transpose(Ztr), Ztr)
        det = da.analyze_determinant(gram)
        # Exercise vector-space and transformation modules on a real standardized row.
        sample = Ztr[0]
        vec_norm = vs.magnitude(sample)
        transform_check = tf.apply_transformation(np.eye(sample.size), sample)
        content = (f"MATRIX REPRESENTATION (manuscript Section 3.1)\n"
          f"X_train shape: {d['X_train'].shape}\nX_test shape: {d['X_test'].shape}\n"
          f"y_train shape: {d['y_train'].shape}\ny_test shape: {d['y_test'].shape}\n"
          f"X_train^T X_train shape: {gram.shape}\n\n"
          f"VECTOR SPACE / RANK (Sections 3.2)\nrank(X_train): {diag['rank_X_train']}\n"
          f"Number of predictors p: {diag['p']}\nLinearly independent columns: {diag['columns_linearly_independent']}\n"
          f"Rank tolerance: {diag['rank_tolerance']:.3e}\n\n"
          f"GRAM MATRIX / DETERMINANT / CONDITIONING (Section 3.4)\n"
          f"G = X_train_std^T X_train_std\n{np.array2string(gram, precision=4, suppress_small=True)}\n"
          f"det(G): {det['determinant']:.6e}\nInvertible by determinant tolerance: {det['is_invertible']}\n"
          f"log|det(G)|: {diag['log_abs_det']:.6f}\ncond(X_train_std): {diag['cond_X_train_std']:.6f}\n"
          f"cond(G): {diag['cond_gram']:.6f}\n\n"
          f"STANDARDIZATION (Section 3.3)\nTraining means: {np.round(d['scaler'].mu, 4)}\n"
          f"Training standard deviations: {np.round(d['scaler'].sigma, 4)}\n"
          f"First raw row: {np.round(d['X_train'][0], 4)}\nFirst standardized row: {np.round(sample, 4)}\n"
          f"Vector magnitude of standardized row: {vec_norm:.6f}\nIdentity transformation check: {np.allclose(transform_check, sample)}\n"
          f"Test data uses the same training scaler: {np.allclose(d['X_test_std'], d['scaler'].transform(d['X_test']))}")
        self._text(self.body, content, 27).pack(fill="both", expand=True)

    def _pca_page(self):
        self._heading("Eigenvalues & PCA", "Eigen-analysis of the training covariance matrix and reduction at the 95% variance threshold.")
        eig = self.analysis["eigen"]
        table = pd.DataFrame({"Eigenvalue": eig["eigenvalues"], "Explained %": eig["explained_ratio"]*100,
                              "Cumulative %": eig["cumulative_ratio"]*100, "Eigenpair residual": eig["residual_norms"]},
                             index=[f"PC{i+1}" for i in range(len(eig["eigenvalues"]))])
        rows = [(idx, *[round(float(v), 6) for v in table.loc[idx]]) for idx in table.index]
        self._tree(self.body, ["pc", "eigen", "explained", "cumulative", "residual"],
                   ["COMPONENT", "EIGENVALUE", "EXPLAINED %", "CUMULATIVE %", "EIGENPAIR RESIDUAL"], rows, 5).pack(fill="x", pady=(0, 10))
        V = eig["eigenvectors"]
        load_rows = [(feature, *[round(float(x), 5) for x in V[i]]) for i, feature in enumerate(dm.FEATURES)]
        self._tree(self.body, ["feature", "pc1", "pc2", "pc3", "pc4"], ["FEATURE", "PC1", "PC2", "PC3", "PC4"], load_rows, 4).pack(fill="x", pady=(0, 10))
        msg = (f"Selected components k: {eig['k_selected']}\n"
               f"Cumulative training variance retained: {eig['cumulative_ratio'][eig['k_selected']-1]*100:.4f}%\n"
               f"Z_train shape: {self.analysis['Z_train'].shape}\nZ_test shape: {self.analysis['Z_test'].shape}\n"
               f"Covariance reconstruction error: {eig['reconstruction_error']:.3e}\n"
               f"Eigenvector orthonormality error: {eig['orthonormal_error']:.3e}\n"
               f"PCA class cross-check selected k: {self.analysis['pca_check'].components.shape[1]}")
        self._text(self.body, msg, 7).pack(fill="both", expand=True)
        tk.Button(self.body, text="Save PCA and exploration plots", bg=BLUE, fg=WHITE, relief="flat", padx=12, pady=7,
                  command=self._save_plots).pack(anchor="w", pady=9)

    def _save_plots(self):
        try:
            eig = self.analysis["eigen"]
            saved = de.save_figures(self.df, eig)
            # Use existing visualization module for the actual-vs-predicted comparison.
            visual = Visualizer(str(OUTPUT_DIR / "plots"))
            saved.append(Path(visual.plot_actual_vs_predicted(
                self.analysis["data"]["y_test"], self.models["Full standardized"]["predictions"],
                model_name="Full Standardized", show=False)))
            messagebox.showinfo("Plots saved", "Saved files:\n" + "\n".join(str(p) for p in saved))
        except Exception as exc: messagebox.showerror("Plot error", str(exc))

    def _regression_page(self):
        self._heading("Regression Analysis", "Controlled comparison using the same 80:20 split and test records.")
        rows = []
        for name, result in self.models.items():
            predictors = "4 standardized features" if name == "Full standardized" else f"{self.analysis['eigen']['k_selected']} principal components"
            rows.append((name, predictors, f"₱{result['rmse']:.2f}", f"{result['r2']:.4f}"))
        self._tree(self.body, ["model", "predictors", "rmse", "r2"], ["MODEL", "PREDICTORS", "TEST RMSE (PHP)", "TEST R²"], rows, 4).pack(fill="x", pady=(0, 12))
        better = min(self.models, key=lambda name: self.models[name]["rmse"])
        info = (f"Training observations: {len(self.analysis['data']['train_df'])}\n"
                f"Testing observations: {len(self.analysis['data']['test_df'])}\n"
                f"Random seed: 42\nRegression solver: least-squares (numpy.linalg.lstsq)\n\n"
                f"Lower test RMSE model: {better}\n\n"
                "Interpretation: RMSE is expressed in PHP. R² may be negative when predictions perform worse than predicting the test-set mean. "
                "Use the measured results as-is; do not claim accuracy beyond these test results.")
        self._text(self.body, info, 12).pack(fill="both", expand=True)

    def _prediction_page(self):
        self._heading("Prediction", "Enter a new student's allowance and spending habits to estimate daily food expense.")
        panel = tk.Frame(self.body, bg=WHITE, highlightbackground=BORDER, highlightthickness=1)
        panel.pack(fill="x", pady=(0, 12))
        fields = [("Daily allowance (PHP)", "175"), ("Meals purchased", "1"),
                  ("Snacks purchased", "1"), ("Campus hours", "7.5")]
        self.prediction_vars = []
        for i, (label, default) in enumerate(fields):
            cell = tk.Frame(panel, bg=WHITE); cell.grid(row=0, column=i, padx=14, pady=16, sticky="ew")
            tk.Label(cell, text=label, bg=WHITE, fg=TEXT, font=("Segoe UI", 9, "bold")).pack(anchor="w")
            var = tk.StringVar(value=default); self.prediction_vars.append(var)
            tk.Entry(cell, textvariable=var, width=17, font=("Segoe UI", 10), relief="solid", bd=1).pack(anchor="w", pady=(7, 0))
            panel.grid_columnconfigure(i, weight=1)
        self.model_choice = tk.StringVar(value="Full standardized")
        choice = tk.Frame(panel, bg=WHITE); choice.grid(row=1, column=0, columnspan=4, sticky="w", padx=14, pady=(0, 12))
        tk.Label(choice, text="Model:", bg=WHITE, fg=TEXT).pack(side="left", padx=(0, 8))
        ttk.Combobox(choice, textvariable=self.model_choice, state="readonly", width=24,
                     values=["Full standardized", "PCA-reduced"]).pack(side="left")
        tk.Button(panel, text="Predict Daily Food Expense", command=self._make_prediction,
                  bg=BLUE, fg=WHITE, relief="flat", padx=14, pady=9).grid(row=1, column=3, padx=14, pady=(0, 12), sticky="e")
        self.prediction_result = tk.Label(self.body, text="Enter the values above and select Predict.",
            bg=WHITE, fg=TEXT, font=("Segoe UI", 12, "bold"), padx=18, pady=18,
            highlightbackground=BORDER, highlightthickness=1)
        self.prediction_result.pack(fill="x", pady=(0, 10))
        tk.Button(self.body, text="Export Prediction History to CSV", command=self._export_predictions,
                  bg="#E8EEF9", fg=TEXT, relief="flat", padx=12, pady=8).pack(anchor="w")

    def _make_prediction(self):
        try:
            values = np.array([float(v.get()) for v in self.prediction_vars], dtype=float)
            if not np.all(np.isfinite(values)): raise ValueError("Enter valid numbers in all fields.")
            if np.any(values < 0): raise ValueError("Values cannot be negative.")
            if values[1] > 2 or values[2] > 4: raise ValueError("Meals must be within observed range (0–2) and snacks within observed range (0–4).")
            if values[3] not in self.df["CampusHours"].unique():
                raise ValueError(f"Campus hours must match an observed survey value: {sorted(self.df['CampusHours'].unique())}.")
            standardized = self.analysis["data"]["scaler"].transform(values.reshape(1, -1))
            name = self.model_choice.get(); model = self.models[name]
            if name == "Full standardized":
                pred_value = prediction.predict_new_sample(model["beta"], *standardized[0])
            else:
                z = standardized @ self.analysis["eigen"]["eigenvectors"][:, :self.analysis["eigen"]["k_selected"]]
                pred_value = float(regression.predict_regression(model["beta"], z)[0])
            pred_value = max(0.0, float(pred_value))
            self.prediction_history.append((values.tolist(), name, pred_value))
            self.prediction_result.config(text=f"Estimated daily food expense: ₱{pred_value:,.2f}\nModel: {name}")
        except Exception as exc: messagebox.showerror("Prediction input error", str(exc))

    def _export_predictions(self):
        if not self.prediction_history:
            messagebox.showinfo("No predictions", "Make at least one prediction before exporting."); return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")],
                                            initialfile="student_food_expense_predictions.csv")
        if not path: return
        try:
            report_export.export_prediction_report(path, [row[2] for row in self.prediction_history])
            messagebox.showinfo("Exported", f"Prediction report saved to:\n{path}")
        except Exception as exc: messagebox.showerror("Export error", str(exc))


if __name__ == "__main__":
    try:
        app = FoodExpenseApp()
        app.mainloop()
    except Exception as error:
        # Display startup errors in a normal traceback as well as a dialog when possible.
        import traceback
        traceback.print_exc()
        try:
            messagebox.showerror("System startup error", str(error))
        except Exception:
            pass
