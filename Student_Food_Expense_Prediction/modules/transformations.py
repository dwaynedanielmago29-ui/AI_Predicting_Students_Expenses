"""Linear and affine transformations for 2D vectors."""
import numpy as np

def scale_vector(vector, sx, sy=None):
    """Scale x and y components; sy defaults to sx (uniform scaling)."""
    if sy is None:
        sy = sx
    v = np.asarray(vector, dtype=float)
    return np.array([sx * v[0], sy * v[1]])

def rotate_vector(vector, angle_degrees):
    """Rotate a 2D vector counterclockwise by angle in degrees."""
    angle = np.deg2rad(angle_degrees)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle),  np.cos(angle)]])
    return rotation @ np.asarray(vector, dtype=float)

def translate_vector(vector, tx, ty):
    """Translate a 2D point/vector using an affine offset."""
    v = np.asarray(vector, dtype=float)
    return v + np.array([tx, ty])

def apply_transformation(matrix, vector):
    """Apply matrix transformation to a vector."""
    return np.asarray(matrix, dtype=float) @ np.asarray(vector, dtype=float)
