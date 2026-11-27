"""Determinant, invertibility and inverse matrix analysis."""
import numpy as np

def determinant(matrix):
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("Determinant requires a square matrix.")
    return float(np.linalg.det(m))

def is_invertible(matrix, tolerance=1e-10):
    return abs(determinant(matrix)) > tolerance

def inverse_matrix(matrix):
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("Inverse requires a square matrix.")
    if not is_invertible(m):
        raise ValueError("Matrix is singular and has no inverse.")
    return np.linalg.inv(m)

def analyze_determinant(matrix):
    det = determinant(matrix)
    return {
        "determinant": det,
        "is_invertible": abs(det) > 1e-10,
        "is_singular": abs(det) <= 1e-10,
    }
