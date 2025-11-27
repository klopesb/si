from unittest import TestCase
import numpy as np
from si.statistics.tanimoto_similarity import tanimoto_similarity


class TestTanimotoSimilarity(TestCase):

    def setUp(self):
        # Test 1: Identical vectors
        self.x1 = np.array([1, 0, 1, 1])
        self.y1 = np.array([[1, 0, 1, 1]])
        
        # Test 2: Multiple samples
        self.x2 = np.array([1, 0, 1, 1])
        self.y2 = np.array([[1, 0, 1, 1],
                           [1, 1, 0, 0],
                           [0, 1, 0, 0]])
        
        # Test 3: Binary example
        self.x3 = np.array([1, 1, 0, 1, 0])
        self.y3 = np.array([[1, 0, 0, 1, 1],
                           [1, 1, 0, 1, 0],
                           [0, 0, 1, 0, 1]])

    def test_identical_vectors(self):
        """Test that identical vectors have similarity of 1.0"""
        result = tanimoto_similarity(self.x1, self.y1)
        self.assertEqual(result[0], 1.0)

    def test_multiple_samples(self):
        """Test similarity with multiple samples"""
        result = tanimoto_similarity(self.x2, self.y2)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], 1.0)  # Identical
        self.assertTrue(0 <= result[1] <= 1)  # Valid similarity
        self.assertTrue(0 <= result[2] <= 1)  # Valid similarity

    def test_binary_example(self):
        """Test with binary examples"""
        result = tanimoto_similarity(self.x3, self.y3)
        self.assertEqual(len(result), 3)
        # Check that all similarities are between 0 and 1
        for similarity in result:
            self.assertTrue(0 <= similarity <= 1)
        # The second sample is identical to x3, so should be 1.0
        self.assertEqual(result[1], 1.0)

    def test_output_shape(self):
        """Test that output shape matches number of samples in y"""
        result = tanimoto_similarity(self.x2, self.y2)
        self.assertEqual(result.shape[0], self.y2.shape[0])

    def test_return_type(self):
        """Test that function returns numpy array"""
        result = tanimoto_similarity(self.x1, self.y1)
        self.assertIsInstance(result, np.ndarray)