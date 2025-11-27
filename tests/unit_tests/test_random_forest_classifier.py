from unittest import TestCase
from datasets import DATASETS_PATH
import os
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.random_forest_classifier import RandomForestClassifier


class TestRandomForestClassifier(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)
        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_fit(self):
        rf = RandomForestClassifier(n_estimators=10)
        rf.fit(self.train_dataset)

        self.assertEqual(rf.n_estimators, 10)
        self.assertEqual(rf.min_sample_split, 2)
        self.assertEqual(rf.max_depth, 10)
        self.assertEqual(len(rf.trees), 10)

    def test_predict(self):
        rf = RandomForestClassifier(n_estimators=10)
        rf.fit(self.train_dataset)
        predictions = rf.predict(self.test_dataset)

        self.assertEqual(predictions.shape[0], self.test_dataset.shape()[0])

    def test_score(self):
        rf = RandomForestClassifier(n_estimators=100, seed=42)
        rf.fit(self.train_dataset)
        accuracy_ = rf.score(self.test_dataset)

        self.assertIsInstance(accuracy_, float)
        self.assertGreaterEqual(accuracy_, 0)
        self.assertLessEqual(accuracy_, 1)
        print(f"Random Forest accuracy on test set: {accuracy_}")