import unittest
import sys
import os
import pathlib

import pandas as pd

PATH_TO_SRC_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), "../", "../", "src/")
sys.path.insert(0, PATH_TO_SRC_DIR)

from learning_algorithms.CART_decision_tree_predictor import RegressionTree

class TestRegressionTree(unittest.TestCase):

    def test_calculate_feature_mse(self):
        cols = ["example", "feature", "class"]
        data = [
            [1, 5,  3], # First group... average class val = 3.4
            [2, 5,  3], # mse = ((3-3.4)^2 * 4 + (5-3.4)^2) / 5 = 0.64
            [3, 5,  5],
            [4, 5,  3],
            [5, 5,  3],
            [6, 10,  8], # Second group... average class val = 8.4
            [7, 10,  8], # mse = ((8-8.4)^2 * 4 + (10-8.4)^2) / 5 = 0.64
            [8, 10,  10],
            [9, 10,  8],
            [10, 10, 8],
        ]
        df = pd.DataFrame(data, columns=cols)
        info_dict = RegressionTree.calculate_feature_mse(df["feature"], df["class"])
        self.assertEqual(7.5, info_dict["feature_split_val"]) # Average of feature values
        self.assertEqual([0, 1, 2, 3, 4], info_dict["lte_feature_split_indicies"])
        self.assertEqual(0.64, round(info_dict["lte_examples_mse"], 2))
        self.assertEqual([5, 6, 7, 8, 9], info_dict["gt_feature_split_indicies"])
        self.assertEqual(0.64, round(info_dict["gt_examples_mse"], 2))
        self.assertEqual(0.64, round(info_dict["avrg_mse"], 2))

    def test_pick_best_feature_to_split(self):
        # Feature 2 results in a split where first group is indicies 0-8 and second group is index 9.
        # This will have much worse MSE than feature 1 where the more similar groups are put togeter. 
        cols = ["example", "feature_1", "feature_2", "class"]
        data = [
            [1, 5, 4, 3], 
            [2, 5, 4,  3],
            [3, 5, 4,  5],
            [4, 5, 4,  3],
            [5, 5, 4,  3],
            [6, 10,  4,  8],
            [7, 10,  4,  8],
            [8, 10,  4,  10],
            [9, 10,  4,  8],
            [10, 10, 100000, 8],
        ]
        df = pd.DataFrame(data, columns=cols)
        (best_feature, feature_split, lte_indicies,
            gte_indicies, lte_mse, gt_mse) = RegressionTree.pick_best_feature_to_split(df.drop("example", axis=1), "class")
        self.assertEqual("feature_1", best_feature)
        self.assertEqual(7.5, feature_split) # Average of feature values
        self.assertEqual([0, 1, 2, 3, 4], lte_indicies)
        self.assertEqual(0.64, round(lte_mse, 2))
        self.assertEqual([5, 6, 7, 8, 9], gte_indicies)
        self.assertEqual(0.64, round(gt_mse, 2))

    def test_build_tree(self):
        cols = ["example", "feature", "class"]
        data = [
            [1, 5,  3], # First group... average class val = 3.4
            [2, 5,  3], # mse = ((3-3.4)^2 * 4 + (5-3.4)^2) / 5 = 0.64
            [3, 5,  5],
            [4, 5,  3],
            [5, 5,  3],
            [6, 10,  8], # Second group... average class val = 8.4
            [7, 10,  8], # mse = ((8-8.4)^2 * 4 + (10-8.4)^2) / 5 = 0.64
            [8, 10,  10],
            [9, 10,  8],
            [10, 10, 8],
        ]
        df = pd.DataFrame(data, columns=cols)
        regression_tree = RegressionTree(df.drop("example", axis=1), "class", 1000)
        regression_tree.build_tree(regression_tree.root)
        self.assertEqual("feature", regression_tree.root.feature)
        self.assertEqual(7.5, regression_tree.root.feature_split_val)
        self.assertEqual(5, len(regression_tree.root.lte_feature_child.data))
        self.assertEqual(5, len(regression_tree.root.gt_feature_child.data))
        self.assertTrue(regression_tree.root.lte_feature_child.can_classify())
        self.assertEqual(3.4, regression_tree.root.lte_feature_child.mean_label())
        self.assertTrue(regression_tree.root.gt_feature_child.can_classify())
        self.assertEqual(8.4, regression_tree.root.gt_feature_child.mean_label())

    def test_build_tree_and_classify(self):
        cols = ["example", "feature", "feature_2", "class"]
        data = [
            [1, 5, 4, 3], 
            [2, 5, 5, 3], 
            [3, 5, 6, 5],
            [4, 5, 5, 3],
            [5, 5, 5, 3],
            [6, 10, 15, 8],
            [7, 10, 16, 8],
            [8, 10, 19, 10],
            [9, 10, 20, 8],
            [10, 10, 25, 8],
        ]
        df = pd.DataFrame(data, columns=cols)
        regression_tree = RegressionTree(df.drop("example", axis=1), "class", 1000)
        regression_tree.build_tree(regression_tree.root)
        for row_index in df.index:
            prediction = regression_tree.classify_example(df.loc[row_index])
            if df.loc[row_index]["class"] < 8:
                self.assertEqual(3.4, prediction)
            else:
                self.assertEqual(8.4, prediction)