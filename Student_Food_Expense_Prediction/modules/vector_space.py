"""Vector arithmetic and vector-space related utilities."""
import numpy as np

def add_vectors(a, b):
    return np.add(a, b)

def subtract_vectors(a, b):
    return np.subtract(a, b)

def dot_product(a, b):
    return float(np.dot(a, b))

def magnitude(vector):
    return float(np.linalg.norm(vector))

def normalize_vector(vector):
    v = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return v / norm

def cosine_similarity(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")
    return float(np.dot(a, b) / denom)
