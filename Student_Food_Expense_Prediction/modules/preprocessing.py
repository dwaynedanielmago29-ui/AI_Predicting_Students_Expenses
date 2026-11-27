"""Data cleaning and feature preprocessing utilities."""
from __future__ import annotations
import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with duplicate rows removed and missing values filled."""
    data = df.copy().drop_duplicates()
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            data[col] = data[col].fillna(data[col].median())
        else:
            mode = data[col].mode(dropna=True)
            data[col] = data[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")
    return data

def encode_categorical(df: pd.DataFrame, drop_first: bool = False) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    return pd.get_dummies(df.copy(), drop_first=drop_first, dtype=int)

def standardize(data):
    """Z-score standardize numeric data; zero-variance columns become zeros."""
    arr = np.asarray(data, dtype=float)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std_safe = np.where(std == 0, 1, std)
    return (arr - mean) / std_safe

def normalize_minmax(data):
    """Scale numeric columns to [0, 1]; constant columns become zeros."""
    arr = np.asarray(data, dtype=float)
    minimum = arr.min(axis=0)
    maximum = arr.max(axis=0)
    span = np.where(maximum - minimum == 0, 1, maximum - minimum)
    return (arr - minimum) / span

def split_features_target(df: pd.DataFrame, target: str):
    """Split a DataFrame into features X and target y."""
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found.")
    return df.drop(columns=[target]), df[target]
