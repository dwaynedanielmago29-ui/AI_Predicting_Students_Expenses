"""Unit tests for Genesis's linear algebra modules."""
import unittest
import numpy as np
from modules.matrix_analysis import matrix_add, matrix_multiply, transpose, matrix_rank
from modules.vector_space import dot_product, magnitude, normalize_vector, cosine_similarity
from modules.transformations import rotate_vector, scale_vector, apply_transformation
from modules.determinant_analysis import determinant, is_invertible, inverse_matrix
from modules.preprocessing import standardize, normalize_minmax

class TestMatrixAnalysis(unittest.TestCase):
    def test_add(self):
        np.testing.assert_array_equal(matrix_add([[1, 2]], [[3, 4]]), [[4, 6]])
    def test_multiply(self):
        np.testing.assert_array_equal(matrix_multiply([[1, 2]], [[3], [4]]), [[11]])
    def test_transpose(self):
        np.testing.assert_array_equal(transpose([[1, 2], [3, 4]]), [[1, 3], [2, 4]])
    def test_rank(self):
        self.assertEqual(matrix_rank([[1, 2], [2, 4]]), 1)

class TestVectorSpace(unittest.TestCase):
    def test_dot_and_magnitude(self):
        self.assertEqual(dot_product([1, 2], [3, 4]), 11)
        self.assertAlmostEqual(magnitude([3, 4]), 5)
    def test_normalize(self):
        np.testing.assert_allclose(normalize_vector([3, 4]), [0.6, 0.8])
    def test_cosine_similarity(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0)

class TestTransformations(unittest.TestCase):
    def test_scale(self):
        np.testing.assert_allclose(scale_vector([2, 3], 2), [4, 6])
    def test_rotate_90(self):
        np.testing.assert_allclose(rotate_vector([1, 0], 90), [0, 1], atol=1e-7)
    def test_apply(self):
        np.testing.assert_array_equal(apply_transformation([[2, 0], [0, 3]], [1, 2]), [2, 6])

class TestDeterminant(unittest.TestCase):
    def test_det(self):
        self.assertAlmostEqual(determinant([[1, 2], [3, 4]]), -2)
    def test_invertible(self):
        self.assertTrue(is_invertible([[1, 2], [3, 4]]))
        self.assertFalse(is_invertible([[1, 2], [2, 4]]))
    def test_inverse(self):
        np.testing.assert_allclose(inverse_matrix([[1, 0], [0, 2]]), [[1, 0], [0, 0.5]])

class TestPreprocessing(unittest.TestCase):
    def test_standardize(self):
        x = standardize([[1, 2], [3, 4]])
        np.testing.assert_allclose(x.mean(axis=0), [0, 0])
    def test_minmax(self):
        np.testing.assert_allclose(normalize_minmax([[1, 2], [3, 4]]), [[0, 0], [1, 1]])

if __name__ == "__main__":
    unittest.main()
