"""Matrix operations and numerical analysis."""
import numpy as np

def create_matrix(data):
    return np.asarray(data, dtype=float)

def matrix_add(a, b):
    return np.add(a, b)

def matrix_subtract(a, b):
    return np.subtract(a, b)

def matrix_multiply(a, b):
    return np.matmul(a, b)

def transpose(matrix):
    return np.asarray(matrix).T

def matrix_rank(matrix):
    return int(np.linalg.matrix_rank(matrix))

def analyze_matrix(matrix):
    m = np.asarray(matrix, dtype=float)
    return {
        "shape": m.shape,
        "rank": int(np.linalg.matrix_rank(m)),
        "is_square": m.ndim == 2 and m.shape[0] == m.shape[1],
        "trace": float(np.trace(m)) if m.ndim == 2 and m.shape[0] == m.shape[1] else None,
    }
