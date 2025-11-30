from unittest import TestCase
from datasets import DATASETS_PATH
import os
from si.io.csv_file import read_csv
import numpy as np
from si.model_selection.split import train_test_split,stratified_train_test_split

class TestSplits(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_train_test_split(self):

        train, test = train_test_split(self.dataset, test_size = 0.2, random_state=123)
        test_samples_size = int(self.dataset.shape()[0] * 0.2)
        self.assertEqual(test.shape()[0], test_samples_size)
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - test_samples_size)

#KB test exercise 6.2
    def test_stratified_train_test_split(self):

        train, test = stratified_train_test_split(self.dataset, test_size=0.2, random_state=123)
        
  
        test_samples_size = int(self.dataset.shape()[0] * 0.2)
        self.assertEqual(test.shape()[0], test_samples_size)
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - test_samples_size)
        

        unique_labels = np.unique(self.dataset.y)
        for label in unique_labels:
            original_count = np.sum(self.dataset.y == label)
            train_count = np.sum(train.y == label)
            test_count = np.sum(test.y == label)
     
            self.assertEqual(original_count, train_count + test_count)