from unittest import TestCase
from datasets import DATASETS_PATH
import os
from si.decomposition.pca import PCA
from si.io.csv_file import read_csv
import numpy as np

#KB test exercise 5

class TestPCA(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        """Test that PCA fit calculates mean, components, and explained variance correctly"""
        n_components = 2
        pca = PCA(n_components=n_components)
        pca.fit(self.dataset)
        
        # Check that mean is calculated and has correct shape
        self.assertIsNotNone(pca.mean)
        self.assertEqual(pca.mean.shape[0], self.dataset.X.shape[1])
        
        # Check that components are calculated and have correct shape
        self.assertIsNotNone(pca.components)
        self.assertEqual(pca.components.shape[0], self.dataset.X.shape[1])
        self.assertEqual(pca.components.shape[1], n_components)
        
        # Check that explained variance is calculated
        self.assertIsNotNone(pca.explained_variance)
        self.assertEqual(len(pca.explained_variance), self.dataset.X.shape[1])
        
        # Check that explained variance is in descending order
        for i in range(len(pca.explained_variance) - 1):
            self.assertGreaterEqual(pca.explained_variance[i], pca.explained_variance[i + 1])
        
        # Check that components are unit vectors (normalized)
        for i in range(pca.components.shape[1]):
            component_norm = np.linalg.norm(pca.components[:, i])
            self.assertAlmostEqual(component_norm, 1.0, places=10)

    def test_transform(self):
        """Test that PCA transform reduces dimensionality correctly"""
        n_components = 2
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(self.dataset)
        
        # Check output shape
        self.assertEqual(X_reduced.shape[0], self.dataset.X.shape[0])
        self.assertEqual(X_reduced.shape[1], n_components)
        
        # Check that output is numpy array
        self.assertIsInstance(X_reduced, np.ndarray)

    def test_fit_transform(self):
        """Test that fit_transform works correctly"""
        n_components = 3
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(self.dataset)
        
        # Check that the result has correct dimensions
        self.assertEqual(X_reduced.shape[0], self.dataset.X.shape[0])
        self.assertEqual(X_reduced.shape[1], n_components)
        
        # Check that all attributes are set after fit_transform
        self.assertIsNotNone(pca.mean)
        self.assertIsNotNone(pca.components)
        self.assertIsNotNone(pca.explained_variance)