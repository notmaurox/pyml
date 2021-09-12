import unittest
import sys
import os
import pathlib

import pandas as pd

PATH_TO_SRC_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), "../", "../", "src/")
sys.path.insert(0, PATH_TO_SRC_DIR)

from data_utils.metrics_evaluator import MetricsEvaluator

class TestMetricsEvaluator(unittest.TestCase):

    def test_calculate_classification_score(self):
        labels = pd.Series(["YEP", "YEP"], index=[0, 1], name='series_1')
        predic = pd.Series(["YEP", "NOPE"], index=[0, 1], name='series_2')
        score = MetricsEvaluator.calculate_classification_score(labels, predic)
        self.assertEqual(score, 0.50)

    def test_calculate_mean_squared_error(self):
        labels = pd.Series([1, 1, 1, 1], index=[0, 1, 2, 3], name='series_1')
        predic = pd.Series([2, 4, 2, 2], index=[0, 1, 2, 3], name='series_2')
        mse = MetricsEvaluator.calculate_mean_squared_error(labels, predic)
        self.assertEqual(mse, 3)

    def test_calculate_precision(self):
        ## less than 1
        labels = pd.Series([1, 0, 1, 1], index=[0, 1, 2, 3], name='series_1')
        predic = pd.Series([1, 1, 1, 1], index=[0, 1, 2, 3], name='series_2')
        precision_class_dict = MetricsEvaluator.calculate_precision(labels, predic)
        # here class label is 1
        self.assertEqual(precision_class_dict[0], None)
        self.assertEqual(precision_class_dict[1], 0.75)

    def test_calculate_recall(self):
        ## less than 1
        labels = pd.Series([1, 1, 1, 1], index=[0, 1, 2, 3], name='series_1')
        predic = pd.Series([1, 0, 1, 1], index=[0, 1, 2, 3], name='series_2')
        recall_class_dict = MetricsEvaluator.calculate_recall(labels, predic)
        # here class label is 1
        self.assertEqual(recall_class_dict[0], None)
        self.assertEqual(recall_class_dict[1], 0.75)
        # second example
        labels = pd.Series([1, 0, 1, 1], index=[0, 1, 2, 3], name='series_1')
        predic = pd.Series([1, 1, 1, 1], index=[0, 1, 2, 3], name='series_2')
        recall_class_dict = MetricsEvaluator.calculate_recall(labels, predic)
        # here class label is 1
        self.assertEqual(recall_class_dict[0], 0)
        self.assertEqual(recall_class_dict[1], 1)

    def test_calculate_f1_score(self):
        ## less than 1
        labels = pd.Series([1, 1, 1, 1], index=[0, 1, 2, 3], name='series_1')
        predic = pd.Series([1, 0, 1, 1], index=[0, 1, 2, 3], name='series_2')
        f1_class_dict = MetricsEvaluator.calculate_f1_score(labels, predic)
        self.assertEqual(f1_class_dict[0], None)
        self.assertEqual(round(f1_class_dict[1], 3), 0.857)

    def test_calculate_calculate_mean_abs_error(self):
        labels = pd.Series([1, 1, 1, 1], index=[0, 1, 2, 3], name='series_1')
        predic = pd.Series([2, 4, 2, 2], index=[0, 1, 2, 3], name='series_2')
        mae = MetricsEvaluator.calculate_mean_abs_error(labels, predic)
        self.assertEqual(mae, 1.5)

    @unittest.skip("todo")
    def test_calculate_r_sqrd_coefficient(self):
        # model always predicts observed - r^2 = Nan
        labels = pd.Series([1, 1, 1],  index=[0, 1, 2], name='series_1')
        predic = pd.Series([1, 1, 1], index=[0, 1, 2], name='series_2')
        mae = MetricsEvaluator.calculate_r_sqrd_coefficient(labels, predic)
        self.assertEqual(mae, None)
        # example 1
        labels = pd.Series([1, 2, 3],  index=[0, 1, 2], name='series_1')
        predic = pd.Series([4, 5, 6], index=[0, 1, 2], name='series_2')
        mae = MetricsEvaluator.calculate_r_sqrd_coefficient(labels, predic)
        self.assertEqual(mae, 1)
        # example 2
        labels = pd.Series([1, 2, 3, 4, 5],  index=[0, 1, 2, 3, 4], name='series_1')
        predic = pd.Series([2, 4, 5, 4, 5], index=[0, 1, 2, 3, 4], name='series_2')
        mae = MetricsEvaluator.calculate_r_sqrd_coefficient(labels, predic)
        self.assertEqual(mae, 0.60)