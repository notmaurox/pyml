import unittest
import sys
import os
import pathlib
import logging

import pandas as pd

PATH_TO_SRC_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), "../", "../", "src/")
sys.path.insert(0, PATH_TO_SRC_DIR)

import learning_algorithms.CART_decision_tree_predictor 
from learning_algorithms.CART_decision_tree_predictor import RegressionTree
from data_utils.data_loader import DataLoader
from data_utils.metrics_evaluator import MetricsEvaluator
from data_utils.data_transformer import DataTransformer

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
LOG.addHandler(handler)

class TestRegressionTree(unittest.TestCase):

    # Provide sample outputs from one test set on one fold for a Regression tree...
    # Show a sample regression tree without early stopping and with early stopping.
    def test_regression_tree_no_early_stopping_vs_early_stopping(self):
        df = DataLoader.load_abalone_data()
        class_col = "rings" #attempt to predict weight of meat
        # Make folds...
        df["class_discretize"] = df[class_col]
        df = DataTransformer.discretize_col(df, "class_discretize", 100)
        folds, _ = DataTransformer.produce_k_fold_cross_validation_sets(
            df, 5, "class_discretize", make_hyperparam_set=False, hyperparam_set_proportion=0.2
        )
        df = df.drop(columns="class_discretize")
        for train_indicies, test_indicies in folds:
            LOG.info(f"Learning on new fold...")
            LOG.info(f"Training on {len(train_indicies)} entitites and testing on {len(test_indicies)} entitites...")
            train_df = df.loc[train_indicies].copy()
            # LOG.info(f"Train set has class representation..\n{train_df[class_col].value_counts()}")
            test_df = df.loc[test_indicies].copy()
            # Build no early stopping tree
            no_early_stop_rt = RegressionTree(train_df, class_col, 0.00)
            no_early_stop_rt.build_tree()
            LOG.info(f"Regression tree without early stopping had {len(no_early_stop_rt.node_store)} nodes")
            classified_tests = no_early_stop_rt.classify_examples(test_df)
            mse_score = MetricsEvaluator.calculate_mean_squared_error(classified_tests[class_col], classified_tests["prediction"])
            LOG.info(f"Tree w/o early stopping had MSE {mse_score}")
            # Build early stopping tree with allowed MSE of 4...
            early_stop_rt = RegressionTree(train_df, class_col, 4)
            early_stop_rt.build_tree()
            LOG.info(f"Regression  tree  with  early stopping had {len(early_stop_rt.node_store)} nodes")
            classified_tests = early_stop_rt.classify_examples(test_df)
            mse_score = MetricsEvaluator.calculate_mean_squared_error(classified_tests[class_col], classified_tests["prediction"])
            LOG.info(f"Tree w   early stopping had MSE {mse_score}")
            # Assure that for the same training set, a tree with early stopping has less nodes than one without
            self.assertLess(len(early_stop_rt.node_store), len(no_early_stop_rt.node_store))
            break

    # Demonstrate the calculation of mean squared error
    def test_calculate_feature_mse(self):
        cols = ["example", "feature", "class"]
        # calculating info required to consider a split for feature "feature" by finding average value in column,
        # splitting into groups >average and <= average, and then classifying each element in each group by the 
        # groups average class value. mse is calculated for both groups. various statistics returned in dict...
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
        # Gets passed down to child node, knows on initialization if it satisfies mse threshold and can stop recursion
        self.assertEqual(0.64, round(info_dict["lte_examples_mse"], 2))
        self.assertEqual([5, 6, 7, 8, 9], info_dict["gt_feature_split_indicies"])
        # Gets passed down to child node, knows on initialization if it satisfies mse threshold and can stop recursion
        self.assertEqual(0.64, round(info_dict["gt_examples_mse"], 2))
        self.assertEqual(0.64, round(info_dict["avrg_mse"], 2))

    # Demonstrate a decision being made to stop growing a subtree (early stopping)
    def test_early_stopping(self):
        learning_algorithms.CART_decision_tree_predictor.handler.setLevel(logging.DEBUG)
        learning_algorithms.CART_decision_tree_predictor.LOG.setLevel(logging.DEBUG)
        # We know that from the above test that each subgroup will have a MSE of 0.64 so setting the early 
        # stopping criteria to mse threshold = 0.64 should result in a tree with 3 nodes...
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
        regression_tree = RegressionTree(df, "class", 0.64)
        regression_tree.build_tree()
        self.assertEqual(3, len(regression_tree.node_store))
        learning_algorithms.CART_decision_tree_predictor.handler.setLevel(logging.INFO)
        learning_algorithms.CART_decision_tree_predictor.LOG.setLevel(logging.INFO)

    # Demonstrate an example traversing a regression tree and a prediction being made at the leaf.
    def test_traversal_and_classification(self):
        learning_algorithms.CART_decision_tree_predictor.handler.setLevel(logging.DEBUG)
        learning_algorithms.CART_decision_tree_predictor.LOG.setLevel(logging.DEBUG)
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
        regression_tree.build_tree()
        for row_index in df.index:
            prediction = regression_tree.classify_example(df.loc[row_index])
            if df.loc[row_index]["class"] < 8:
                self.assertEqual(3.4, prediction)
            else:
                self.assertEqual(8.4, prediction)
        learning_algorithms.CART_decision_tree_predictor.handler.setLevel(logging.INFO)
        learning_algorithms.CART_decision_tree_predictor.LOG.setLevel(logging.INFO)

    # Test picking the best feature to split...
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

    # Test building of a tree...
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
        regression_tree.build_tree()
        self.assertEqual("feature", regression_tree.root.feature)
        self.assertEqual(7.5, regression_tree.root.feature_split_val)
        self.assertEqual(5, len(regression_tree.root.lte_feature_child.data))
        self.assertEqual(5, len(regression_tree.root.gt_feature_child.data))
        self.assertTrue(regression_tree.root.lte_feature_child.can_classify())
        self.assertEqual(3.4, regression_tree.root.lte_feature_child.mean_label())
        self.assertTrue(regression_tree.root.gt_feature_child.can_classify())
        self.assertEqual(8.4, regression_tree.root.gt_feature_child.mean_label())