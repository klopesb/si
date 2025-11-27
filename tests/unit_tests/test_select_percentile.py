from unittest import TestCase
from datasets import DATASETS_PATH
from si.data.dataset import Dataset

import os
from si.feature_selection.select_percentile import SelectPercentile
from si.statistics.f_classification import f_classification
from si.io.data_file import read_data_file


class TestSelectPercentile(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

    def test_fit(self):
        select_percentile = SelectPercentile(score_func=f_classification, percentile=40)

        select_percentile.fit(self.dataset)

        # Testa se F-scores e p-values foram calculados
        self.assertIsNotNone(select_percentile.F)
        self.assertIsNotNone(select_percentile.p)
        self.assertTrue(select_percentile.F.shape[0] > 0)
        self.assertTrue(select_percentile.p.shape[0] > 0)

    def test_transform(self):
        select_percentile = SelectPercentile(score_func=f_classification, percentile=40)

        select_percentile.fit(self.dataset)
        new_dataset = select_percentile.transform(self.dataset)

        # Verifica que o número de features diminuiu ou ficou igual
        self.assertLessEqual(len(new_dataset.features), len(self.dataset.features))

        # Verifica que o número de colunas de X diminuiu ou ficou igual
        self.assertLessEqual(new_dataset.X.shape[1], self.dataset.X.shape[1])

        # Verifica consistência: nº features == nº colunas
        self.assertEqual(len(new_dataset.features), new_dataset.X.shape[1])
