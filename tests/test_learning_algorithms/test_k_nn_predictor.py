import unittest
import sys
import os
import pathlib

import pandas as pd

PATH_TO_SRC_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), "../", "../", "src/")
sys.path.insert(0, PATH_TO_SRC_DIR)

from learning_algorithms.k_nn_predictor import KNearestNeighborPredictor

class KNearestNeighborsPredictor(unittest.TestCase):

    def test_k_nearest_neighbor_classification(self):
        knnp = KNearestNeighborPredictor(
            4, False
        )
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
        pred_df = knnp.k_nearest_neighbor("class", train_df, test_df)
        print(pred_df)
        self.assertTrue(pd.Series(["c1", "c2", "c3"]).equals(pred_df["prediction"]))

    def test_k_nearest_neighbor_classification_regression(self):
        knnp = KNearestNeighborPredictor(
            2, True
        )
        # train set
        cols = ["class", "int_field", "int_field"]
        data = [
            [1, 1, 1], #0
            [2, 2, 2], #1
            [3, 3, 3], #2
            [1, 1, 1], #0
            [2, 2, 2], #1
            [3, 3, 3], #2
        ]
        train_df = pd.DataFrame(data, columns=cols)
        # test set
        cols = ["class", "int_field", "int_field"]
        data = [
            [1, 1, 1], #0
            [2, 2, 2], #1
            [3, 3, 3], #2
        ]
        test_df = pd.DataFrame(data, columns=cols)
        # run pred
        pred_df = knnp.k_nearest_neighbor("class", train_df, test_df)
        print(pred_df)
        self.assertTrue(pd.Series([1.0, 2.0, 3.0]).equals(pred_df["prediction"]))

    def test_make_edited_k_nn_train_set_rmv_misclassified(self):
        knnp = KNearestNeighborPredictor(
            1, False
        )
        # train set
        cols = ["class", "int_field", "int_field"]
        data = [
            ["c1", 1, 1], #0
            ["c2", 2, 2], #1
            ["c3", 3, 3], #2
        ]
        train_df = pd.DataFrame(data, columns=cols)
        df_indicies = knnp.make_edited_k_nn_train_set(
            "class",
            train_df,
        )
        # only one item from each class should end up in the train set...
        print(df_indicies)
        self.assertEqual(0, len(df_indicies))

    def test_make_edited_k_nn_train_set_rmv_misclassified_regression(self):
        knnp = KNearestNeighborPredictor(
            1, True, sigma=2.5, allowed_error=1
        )
        # train set
        cols = ["class", "int_field", "int_field"]
        data = [
            [1, 1, 1], #0
            [2, 2, 2], #1
            [3, 3, 3], #2
        ]
        train_df = pd.DataFrame(data, columns=cols)
        df_indicies = knnp.make_edited_k_nn_train_set(
            "class",
            train_df,
        )
        # only one item from each class should end up in the train set...
        print(df_indicies)
        self.assertEqual(3, len(df_indicies))

    def test_make_edited_k_nn_train_set_rmv_misclassified_empty(self):
        knnp = KNearestNeighborPredictor(
            1, False
        )
        # train set
        cols = ["class", "int_field", "int_field"]
        data = [
            ["c1", 1, 1], #0
            ["c2", 2, 2], #1
            ["c3", 3, 3], #2
        ]
        train_df = pd.DataFrame(data, columns=cols)
        df_indicies = knnp.make_edited_k_nn_train_set(
            "class",
            train_df,
        )
        # only one item from each class should end up in the train set...
        print(df_indicies)
        self.assertEqual(0, len(df_indicies))

    def test_make_edited_k_nn_train_set_rmv_misclassified(self):
        knnp = KNearestNeighborPredictor(
            1, False
        )
        # train set
        cols = ["class", "int_field", "int_field"]
        data = [
            ["c1", 1, 1], #0
            ["c2", 2, 2], #1
            ["c3", 3, 3], #2
            ["c1", 1, 1], #0
            ["c2", 2, 2], #1
            ["c3", 3, 3], #2
            ["c4", 1, 1], #0 - will be missclassified
            ["c5", 2, 2], #1 - will be missclassified
            ["c6", 3, 3], #2 - will be missclassified
        ]
        train_df = pd.DataFrame(data, columns=cols)
        df_indicies = knnp.make_edited_k_nn_train_set(
            "class",
            train_df,
        )
        # only one item from each class should end up in the train set...
        print(df_indicies)
        self.assertEqual(6, len(df_indicies))

    def test_make_edited_k_nn_train_set_rmv_correctly_classified(self):
        knnp = KNearestNeighborPredictor(
            1, False
        )
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
        print(train_df)
        df_indicies = knnp.make_edited_k_nn_train_set(
            "class",
            train_df,
            remove_correct=True
        )
        # only one item from each class should end up in the train set...
        print(df_indicies)
        self.assertEqual(3, len(df_indicies))

    def test_edited_k_nearest_neighbor(self):
        knnp = KNearestNeighborPredictor(
            1, False
        )
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
        pred_df = knnp.edited_k_nearest_neighbor(
            "class",
            train_df,
            test_df,
            remove_correct=True
        )
        print(pred_df)
        print(knnp.training_set_sizes)
        print(knnp.classification_scores)
        self.assertTrue(pd.Series(["c1", "c2", "c3"]).equals(pred_df["prediction"]))

    def test_edited_k_nearest_neighbor_regression(self):
        knnp = KNearestNeighborPredictor(
            2, True, sigma=2.5, allowed_error=0
        )
        # train set
        cols = ["class", "int_field", "int_field"]
        data = [
            [1, 1, 1], #0
            [2, 2, 2], #1
            [3, 3, 3], #2
            [1, 1, 1], #0
            [2, 2, 2], #1
            [3, 3, 3], #2
        ]
        train_df = pd.DataFrame(data, columns=cols)
        # test set
        cols = ["class", "int_field", "int_field"]
        data = [
            [1, 1, 1], #0
            [2, 2, 2], #1
            [3, 3, 3], #2
        ]
        test_df = pd.DataFrame(data, columns=cols)
        # run pred
        pred_df = knnp.edited_k_nearest_neighbor(
            "class",
            train_df,
            test_df,
            remove_correct=True
        )
        print(pred_df)
        print(knnp.training_set_sizes)
        print(knnp.classification_scores)
        self.assertTrue(pd.Series([1.0, 2.0, 3.0]).equals(pred_df["prediction"]))

    def test_make_condensed_k_nearest_neighbor_train_set(self):
        knnp = KNearestNeighborPredictor(
            None, False
        )
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
        df_indicies = knnp.make_condensed_k_nn_train_set(
            "class",
            train_df,
        )
        # only one item from each class should end up in the train set...
        print(df_indicies)
        self.assertEqual(3, len(df_indicies))

    def test_make_condensed_k_nearest_neighbor_train_set_err(self):
        knnp = KNearestNeighborPredictor(
            None, False, 0.5
        )
        # train set
        cols = ["class", "int_field", "int_field"]
        data = [
            [1, 1, 1], #0
            [2, 2, 2], #1
            [3, 3, 3], #2
            [1, 1, 1], #0
            [2, 2, 2], #1
            [3, 3, 3], #2
        ]
        train_df = pd.DataFrame(data, columns=cols)
        df_indicies = knnp.make_condensed_k_nn_train_set(
            "class",
            train_df
        )
        # only one item from each class should end up in the train set...
        print(df_indicies)
        self.assertEqual(3, len(df_indicies))
        knnp.allowed_error = 3
        df_indicies = knnp.make_condensed_k_nn_train_set(
            "class",
            train_df
        )
        # only one item from each class should end up in the train set...
        print(df_indicies)
        self.assertEqual(1, len(df_indicies))

    def test_condensed_k_nearest_neighbor(self):
        knnp = KNearestNeighborPredictor(
            1, False
        )
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
        pred_df = knnp.condensed_k_nearest_neighbor(
            "class",
            train_df,
            test_df
        )
        print(pred_df)
        print(knnp.training_set_sizes)
        print(knnp.classification_scores)
        self.assertTrue(pd.Series(["c1", "c2", "c3"]).equals(pred_df["prediction"]))

    def test_condensed_k_nearest_neighbor_regression(self):
        knnp = KNearestNeighborPredictor(
            1, True, sigma=1, allowed_error=0.00
        )
        # train set
        cols = ["class", "int_field", "int_field"]
        data = [
            [1, 1, 1], #0
            [2, 2, 2], #1
            [3, 3, 3], #2
            [4, 1, 1], #0
            [5, 2, 2], #1
            [6, 3, 3], #2
        ]
        train_df = pd.DataFrame(data, columns=cols)
        # test set
        cols = ["class", "int_field", "int_field"]
        data = [
            [1, 1, 1], #0
            [2, 2, 2], #1
            [3, 3, 3], #2
        ]
        test_df = pd.DataFrame(data, columns=cols)
        # run pred
        pred_df = knnp.condensed_k_nearest_neighbor(
            "class",
            train_df,
            test_df
        )
        print(pred_df)
        print(knnp.training_set_sizes)
        print(knnp.classification_scores)
        self.assertTrue(pd.Series([1.0, 2.0, 3.0]).equals(pred_df["prediction"]))