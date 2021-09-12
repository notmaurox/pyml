import unittest
import sys
import os
import pathlib

import pandas as pd

PATH_TO_SRC_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), "../", "../", "src/")
sys.path.insert(0, PATH_TO_SRC_DIR)

from learning_algorithms.majority_predictor import MajorityPredictor

class TestDataTransformer(unittest.TestCase):

    def test_majority_predictor_simple(self):
        # train set
        cols = ["sample", "str_field"]
        data = [
            ["train_1", "red"], #0
            ["train_2", "red"], #1
            ["train_3", "red"], #2
            ["train_4", "red"], #3
            ["train_5", "red"], #4
            ["train_6", "red"], #5
            ["train_7", "grn"], #6
            ["train_8", "grn"], #7
            ["train_9", "blu"], #8
            ["train_10", "blu"], #9
        ]
        train_df = pd.DataFrame(data, columns=cols)
        # test set
        cols = ["sample", "str_field"]
        data = [
            ["test_1", "red"], #0
            ["test_2", "blu"], #1
            ["test_3", "grn"], #2
        ]
        test_df = pd.DataFrame(data, columns=cols)
        pred_df = MajorityPredictor.predict_by_majority(
            "str_field",
            train_df,
            test_df
        )
        self.assertTrue(
            pred_df["predicted_class"].equals(    
                pd.Series(["red", "red", "red"],  index=[0, 1, 2], name='pred_labels')
            )
        )