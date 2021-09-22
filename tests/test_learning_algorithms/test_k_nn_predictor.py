import unittest
import sys
import os
import pathlib

import pandas as pd

PATH_TO_SRC_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), "../", "../", "src/")
sys.path.insert(0, PATH_TO_SRC_DIR)

from learning_algorithms.k_nn_predictor import KNearestNeighborPredictor

class KNearestNeighborsPredictor(unittest.TestCase):

    def test_majority_predictor_simple(self):
        # train set
        cols = ["class", "int_field", "int_field"]
        data = [
            ["c1", 1, 1], #0
            ["c2", 2, 2], #1
            ["c3", 3, 3], #2
            ["c1", 1, 1], #0
            ["c2", 2, 2], #1
            ["c3", 3, 3], #2
        ]
        train_df = pd.DataFrame(data, columns=cols)
        # test set
        cols = ["class", "int_field", "int_field"]
        data = [
            ["c1", 1, 1], #0
            ["c2", 2, 2], #1
            ["c3", 3, 3], #2
        ]
        test_df = pd.DataFrame(data, columns=cols)
        # run pred
        pred_df = KNearestNeighborPredictor.k_nearest_neighbor(
            4,
            "class",
            train_df,
            test_df
        )
        print(pred_df)
        self.assertTrue(pd.Series(["c1", "c2", "c3"]).equals(pred_df["predicted_class"]))