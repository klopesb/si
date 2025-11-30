from unittest import TestCase
from datasets import DATASETS_PATH
import os
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.ridge_regression_least_squares import RidgeRegressionLeastSquares

#KB test exercise 8

class TestRidgeRegressionLeastSquares(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)
        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_fit(self):
        ridge = RidgeRegressionLeastSquares()
        ridge.fit(self.train_dataset)

        self.assertEqual(ridge.theta.shape[0], self.train_dataset.shape()[1])
        self.assertIsNotNone(ridge.theta_zero)
        self.assertIsNotNone(ridge.mean)
        self.assertIsNotNone(ridge.std)

    def test_predict(self):
        ridge = RidgeRegressionLeastSquares()
        ridge.fit(self.train_dataset)
        predictions = ridge.predict(self.test_dataset)

        self.assertEqual(predictions.shape[0], self.test_dataset.shape()[0])

    def test_score(self):
        ridge = RidgeRegressionLeastSquares()
        ridge.fit(self.train_dataset)
        mse_ = ridge.score(self.test_dataset)

        self.assertIsInstance(mse_, float)
        self.assertGreaterEqual(mse_, 0)