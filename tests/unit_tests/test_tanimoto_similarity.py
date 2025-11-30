from unittest import TestCase
import numpy as np
from si.statistics.tanimoto_similarity import tanimoto_similarity

#KB test exercise 4

class TestTanimotoSimilarity(TestCase):

    def setUp(self):
        self.x1 = np.array([1, 0, 1, 1])
        self.y1 = np.array([[1, 0, 1, 1]])
        

    def test_identical_vectors(self):
        result = tanimoto_similarity(self.x1, self.y1)
        self.assertEqual(result[0], 1.0)
