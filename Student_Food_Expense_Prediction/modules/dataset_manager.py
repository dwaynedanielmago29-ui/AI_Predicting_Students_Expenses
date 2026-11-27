"""
dataset_manager.py
Dataset Management module (Andrey Arellano)

Project : Predicting Students' Daily Food Expenses Based on Their Spending
          Habits and Allowance Using Linear Algebra-Based Regression Modeling
Course  : AI 101 - IT 2101 (Linear Algebra for AI | Final Project)

This module supports the manuscript sections on data collection, cleaning,
split protocol, and matrix representation:

    Section 2.1  Data collection, cleaning and inclusion criteria
                 - 100 raw survey responses, 2 incomplete -> 98 complete (n = 98)
                 - range answers converted to representative midpoints
    Section 2.2  Split protocol
                 - 80:20 train/test split, fixed random seed 42 (78 train / 20 test)
                 - mean and standard deviation learned from the TRAINING set only
    Section 3.1  Matrix representation  X in R^(n x p), y in R^n
    Section 3.3  Standardization as z = (x - mu) M,  M = diag(1/sigma_1 ... 1/sigma_p)

Dataset source: researcher-collected Google Forms survey (https://shorturl.at/taBXB)
Permitted use : academic, non-commercial class research only; anonymized data.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and constants (taken from the manuscript)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_READY_PATH = DATA_DIR / "Student_Daily_Food_Expenses_Model_Ready.csv"
ORIGINAL_PATH = DATA_DIR / "original_responses.csv"

ID_COLUMN = "RespondentID"
TARGET = "DailyFoodExpense_PHP"
FEATURES = [
    "DailyAllowance_PHP",
    "MealsPurchased",
    "SnacksPurchased",
    "CampusHours",
]
NUMERIC_COLUMNS = FEATURES + [TARGET]

RANDOM_SEED = 42        # Section 2.2
TEST_SIZE = 0.20        # Section 2.2 (80:20)
EXPECTED_RAW_N = 100    # original survey responses
EXPECTED_CLEAN_N = 98   # complete responses used for modeling


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_model_ready(path=MODEL_READY_PATH):
    """Load the cleaned, model-ready dataset (98 complete responses)."""
    path = Path(path)
    if not path.exists() and path == MODEL_READY_PATH:
        alternate = DATA_DIR / "Student_Daily_Food_Expenses_Model_Ready (1).csv"
        if alternate.exists():
            path = alternate
    if not path.exists():
        raise FileNotFoundError(f"Model-ready dataset not found: {path}")
    return pd.read_csv(path)


def load_original(path=ORIGINAL_PATH):
    """Load the raw survey responses (empty DataFrame if the file has no rows)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Original responses file not found: {path}")
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Cleaning raw survey responses (Section 2.1)
# ---------------------------------------------------------------------------
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def range_to_midpoint(value):
    """
    Convert a survey answer to one representative number.

        "100-150"      -> 125.0   (midpoint of the range)
        "PHP 51 - 100" -> 75.5    (midpoint of the range)
        "175.5", 3     -> 175.5, 3.0
        anything else  -> NaN

    Midpoints are approximations, not exact amounts (see Assumptions).
    """
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).replace(",", "")
    numbers = [float(n) for n in _NUMBER.findall(text)]
    if len(numbers) == 1:
        return numbers[0]
    if len(numbers) == 2:
        return (numbers[0] + numbers[1]) / 2
    return np.nan


def clean_original_responses(raw, column_map=None):
    """
    Turn raw survey responses into the model-ready format.

    column_map : optional {raw_column_name: model_ready_column_name} when the
                 Google Forms headers differ from the model-ready names.

    Steps (Section 2.1):
        1. strip header whitespace and rename columns
        2. convert range answers to midpoints / numbers
        3. remove exact duplicate responses
        4. EXCLUDE responses with missing required values (no imputation)
        5. keep only the four predictors, the target and the ID
           (personal information is not carried into the modeling dataset)

    Returns (clean_df, log) where log records how many rows were removed.
    """
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if column_map:
        df = df.rename(columns=column_map)

    missing_cols = [c for c in NUMERIC_COLUMNS if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Required columns not found in raw data: {missing_cols}")

    log = {"raw_rows": int(len(df))}

    for col in NUMERIC_COLUMNS:
        df[col] = df[col].map(range_to_midpoint)

    before = len(df)
    df = df.drop_duplicates()
    log["duplicates_removed"] = int(before - len(df))

    before = len(df)
    df = df.dropna(subset=NUMERIC_COLUMNS)
    log["incomplete_removed"] = int(before - len(df))

    df = df.reset_index(drop=True)
    if ID_COLUMN not in df.columns:
        df.insert(0, ID_COLUMN, [f"S{i:03d}" for i in range(1, len(df) + 1)])

    clean = df[[ID_COLUMN] + NUMERIC_COLUMNS]
    log["clean_rows"] = int(len(clean))
    return clean, log


def save_dataset(df, path=MODEL_READY_PATH):
    """Save a DataFrame to CSV without the index."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return Path(path)


# ---------------------------------------------------------------------------
# Validation (Section 2.1: missing, duplicate, inconsistent, unusual values)
# ---------------------------------------------------------------------------
def validate_dataset(df):
    """Check the dataset and return a dictionary of results."""
    expected = [ID_COLUMN] + NUMERIC_COLUMNS
    missing_columns = [c for c in expected if c not in df.columns]
    present = [c for c in NUMERIC_COLUMNS if c in df.columns]
    non_numeric = [c for c in present if not pd.api.types.is_numeric_dtype(df[c])]
    numeric_ok = [c for c in present if c not in non_numeric]

    negatives = {c: int((df[c] < 0).sum()) for c in numeric_ok if (df[c] < 0).any()}
    dup_ids = int(df[ID_COLUMN].duplicated().sum()) if ID_COLUMN in df.columns else 0
    zero_var = [c for c in numeric_ok if df[c].nunique() <= 1]

    report = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "matches_expected_n": len(df) == EXPECTED_CLEAN_N,
        "missing_columns": missing_columns,
        "missing_percent": (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicate_ids": dup_ids,
        "duplicate_rows": int(df.duplicated().sum()),
        "negative_values": negatives,
        "non_numeric_columns": non_numeric,
        "zero_variance_columns": zero_var,
    }
    report["is_valid"] = bool(
        not missing_columns and not non_numeric and not negatives
        and dup_ids == 0 and not zero_var
        and df[present].isnull().sum().sum() == 0
    )
    return report


def print_validation_report(report):
    """Print the result of validate_dataset()."""
    print("=" * 64)
    print("DATASET VALIDATION REPORT")
    print("=" * 64)
    print(f"Rows: {report['rows']} (expected {EXPECTED_CLEAN_N}: "
          f"{'OK' if report['matches_expected_n'] else 'DIFFERENT'})   "
          f"Columns: {report['columns']}")
    print(f"Missing required columns : {report['missing_columns'] or 'None'}")
    print(f"Non-numeric columns      : {report['non_numeric_columns'] or 'None'}")
    print(f"Negative values          : {report['negative_values'] or 'None'}")
    print(f"Zero-variance columns    : {report['zero_variance_columns'] or 'None'}")
    print(f"Duplicate RespondentIDs  : {report['duplicate_ids']}")
    print(f"Duplicate rows           : {report['duplicate_rows']}")
    print("Missing values (%):")
    for col, pct in report["missing_percent"].items():
        print(f"    {col:22s} {pct:6.2f}")
    print(f"Dataset valid: {'YES' if report['is_valid'] else 'NO'}")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Split protocol (Section 2.2)
# ---------------------------------------------------------------------------
def train_test_split(df, test_size=TEST_SIZE, seed=RANDOM_SEED):
    """
    Reproducible shuffle-and-split. With n = 98, test_size = 0.20 and seed 42
    this gives 78 training rows and 20 testing rows.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test = int(round(len(df) * test_size))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return (df.iloc[train_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True))


# ---------------------------------------------------------------------------
# Matrix representation (Section 3.1)
# ---------------------------------------------------------------------------
def get_X_y(df, features=None, add_intercept=False):
    """
    Return X in R^(n x p) and y in R^n. Each row is one student, each column
    one feature. With add_intercept=True a leading column of ones gives X_b
    (used for the regression in Section 4.1).
    """
    features = features or FEATURES
    X = df[features].to_numpy(dtype=float)
    if add_intercept:
        X = np.column_stack([np.ones(len(X)), X])
    y = df[TARGET].to_numpy(dtype=float)
    return X, y


def gram_matrix(X):
    """Gram matrix G = X^T X (p x p)."""
    return X.T @ X


def rank_one_update_check(X, x_new):
    """
    Verify the rank-one update from Section 3.1:
        X_new^T X_new = X^T X + x x^T
    where X_new is X with one extra row x^T (given consistent preprocessing).
    Returns the maximum absolute difference between both sides (about 0).
    """
    x_new = np.asarray(x_new, dtype=float).reshape(1, -1)
    X_new = np.vstack([X, x_new])
    lhs = X_new.T @ X_new
    rhs = X.T @ X + x_new.T @ x_new
    return float(np.max(np.abs(lhs - rhs)))


# ---------------------------------------------------------------------------
# Standardization as an affine map (Section 3.3)
# ---------------------------------------------------------------------------
class Standardizer:
    """
    z = (x - mu) M   with   M = diag(1/sigma_1, ..., 1/sigma_p)  (row-vector form)

    mu and sigma are learned from the TRAINING data only and re-used,
    unchanged, on the testing data (prevents data leakage).
    Sample standard deviation (ddof = 1) is used, so that cov(Z) is the
    correlation matrix of the training features.
    """

    def __init__(self, ddof=1):
        self.ddof = ddof
        self.mu = None
        self.sigma = None
        self.M = None

    def fit(self, X_train):
        X_train = np.asarray(X_train, dtype=float)
        self.mu = X_train.mean(axis=0)
        self.sigma = X_train.std(axis=0, ddof=self.ddof)
        if np.any(self.sigma == 0):
            bad = np.where(self.sigma == 0)[0].tolist()
            raise ValueError(f"Zero-variance feature(s) at column index {bad}")
        self.M = np.diag(1.0 / self.sigma)
        return self

    def transform(self, X):
        self._check_fitted()
        return (np.asarray(X, dtype=float) - self.mu) @ self.M

    def fit_transform(self, X_train):
        return self.fit(X_train).transform(X_train)

    def inverse_transform(self, Z):
        """Undo the transformation: x = z M^-1 + mu."""
        self._check_fitted()
        return np.asarray(Z, dtype=float) @ np.linalg.inv(self.M) + self.mu

    def _check_fitted(self):
        if self.mu is None:
            raise RuntimeError("Standardizer has not been fitted yet")


# ---------------------------------------------------------------------------
# One call that prepares everything the other modules need
# ---------------------------------------------------------------------------
def prepare_data(df=None, test_size=TEST_SIZE, seed=RANDOM_SEED):
    """
    Split, build matrices and standardize using training parameters only.

    Returns a dictionary with train_df, test_df, X_train, X_test, y_train,
    y_test, X_train_std, X_test_std and the fitted Standardizer ('scaler').
    """
    df = load_model_ready() if df is None else df
    train_df, test_df = train_test_split(df, test_size, seed)
    X_train, y_train = get_X_y(train_df)
    X_test, y_test = get_X_y(test_df)

    scaler = Standardizer().fit(X_train)
    return {
        "train_df": train_df, "test_df": test_df,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_std": scaler.transform(X_train),
        "X_test_std": scaler.transform(X_test),
        "scaler": scaler,
    }


def print_matrix_evidence(data):
    """Print the evidence listed in Sections 3.1 and 3.3 of the manuscript."""
    Xtr, Xte = data["X_train"], data["X_test"]
    Ztr, scaler = data["X_train_std"], data["scaler"]

    print("\nMATRIX REPRESENTATION (Section 3.1)")
    print(f"X_train shape : {Xtr.shape}   X_test shape : {Xte.shape}")
    print(f"y_train shape : {data['y_train'].shape}   y_test shape : {data['y_test'].shape}")
    G = gram_matrix(Ztr)
    print(f"Gram matrix (standardized train) shape : {G.shape}")
    print(f"Off-diagonal entry G[{FEATURES[0]}, {FEATURES[1]}] = {G[0, 1]:.4f}")
    diff = rank_one_update_check(Ztr, data["X_test_std"][0])
    print(f"Rank-one update check, max |lhs - rhs| = {diff:.2e}")

    print("\nSTANDARDIZATION (Section 3.3)  z = (x - mu) M")
    for name, m, s in zip(FEATURES, scaler.mu, scaler.sigma):
        print(f"    {name:22s} mu = {m:9.4f}   sigma = {s:8.4f}   1/sigma = {1/s:8.4f}")
    x0, z0 = Xtr[0], Ztr[0]
    print(f"Raw sample x : {np.round(x0, 3)}")
    print(f"Transformed z: {np.round(z0, 3)}")
    back = scaler.inverse_transform(z0)
    print(f"Inverse check, max |x - x_back| = {np.max(np.abs(x0 - back)):.2e}")
    print(f"Test set uses the train mu/sigma (not refitted): "
          f"{np.allclose(data['X_test_std'], scaler.transform(Xte))}")


# ---------------------------------------------------------------------------
# Run as a script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model_ready = load_model_ready()
    print_validation_report(validate_dataset(model_ready))

    prepared = prepare_data(model_ready)
    print(f"\nSplit (seed {RANDOM_SEED}): train = {len(prepared['train_df'])}, "
          f"test = {len(prepared['test_df'])}")
    print_matrix_evidence(prepared)

    original = load_original()
    if original.empty:
        print("\noriginal_responses.csv has no rows yet; nothing to clean. "
              "Paste the raw Google Forms export there to rebuild the model-ready file.")
    else:
        clean, log = clean_original_responses(original)
        print(f"\nCleaning log for original_responses.csv: {log}")
