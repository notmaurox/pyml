import unittest
import sys
import os
import pathlib

import pandas as pd

PATH_TO_SRC_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), "../", "../", "src/")
sys.path.insert(0, PATH_TO_SRC_DIR)

from learning_algorithms.k_nn_predictor import KNearestNeighborPredictor

class TestKNearestNeighborsPredictor(unittest.TestCase):

    # Demonstrate the calculation of your distance function
    def test_helper_fx_find_datapoint_k_neighbors_distance(self):
        knnp = KNearestNeighborPredictor(
            3, False
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
            ["c1", 1, 1], #0
        ]
        train_df = pd.DataFrame(data, columns=cols)
        # test set
        cols = ["class", "int_field", "int_field"]
        data = [
            ["c1", 1, 1], #0
        ]
        test_df = pd.DataFrame(data, columns=cols)
        neighbors = knnp._find_datapoint_k_neighbors(0, "class", train_df, test_df)
        self.assertEqual(len(neighbors), 3)
        self.assertEqual([0, 0, 0], [neighbor[0] for neighbor in neighbors])
        self.assertEqual(["c1", "c1", "c1"], [neighbor[1] for neighbor in neighbors])

    # Demonstrate the calculation of your kernel function
    def test_helper_fx_find_datapoint_k_neighbors_gaussian_kernel(self):
        knnp = KNearestNeighborPredictor(
            3, True
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
            ["c1", 1, 1], #0
        ]
        train_df = pd.DataFrame(data, columns=cols)
        # test set
        cols = ["class", "int_field", "int_field"]
        data = [
            ["c1", 1, 1], #0
        ]
        test_df = pd.DataFrame(data, columns=cols)
        neighbors = knnp._find_datapoint_k_neighbors(0, "class", train_df, test_df)
        self.assertEqual(len(neighbors), 3)
        self.assertEqual([1.0, 1.0, 1.0], [neighbor[0] for neighbor in neighbors])
        self.assertEqual(["c1", "c1", "c1"], [neighbor[1] for neighbor in neighbors])

    # Demonstrate an example of a point being classified using k-nn. Show the neighbors returned as well as the point
    # being classified.
    def test_k_nn_classification_step_wise(self):
        knnp = KNearestNeighborPredictor(
            5, False
        )
        # train set
        cols = ["class", "int_field", "int_field"]
        data = [
            ["c1", 1, 1], #0
            ["c2", 2, 2], #1 - dist from c1 = sqrt(1+1) = 1.4142135623730951
            ["c3", 3, 3], #2 - dist from c1 = sqrt(4+4) = 2.8284271247461903
            ["c1", 1, 1], #0
            ["c2", 2, 2], #1
            ["c3", 3, 3], #2
            ["c1", 1, 1], #0
        ]
        train_df = pd.DataFrame(data, columns=cols)
        # test set
        cols = ["class", "int_field", "int_field"]
        data = [
            ["c1", 1, 1], #0
        ]
        test_df = pd.DataFrame(data, columns=cols)
        neighbors = knnp._find_datapoint_k_neighbors(0, "class", train_df, test_df)
        self.assertEqual(len(neighbors), 5)
        self.assertEqual([0, 0, 0, 1.4142135623730951, 2.8284271247461903], sorted([neighbor[0] for neighbor in neighbors]))
        self.assertEqual(["c1", "c1", "c1", "c2", "c3"], sorted([neighbor[1] for neighbor in neighbors]))
        # Classification by majority neghbor label
        prediction = knnp._classify_point_from_neighbors(neighbors)
        self.assertEqual(prediction, "c1")

    # Demonstrate an example of a point being regressed using k-nn. Show the neighbors returned as
    # well as the point being predicted.
    def test_k_nn_regression_step_wise(self):
        knnp = KNearestNeighborPredictor(
            5, True, sigma=0.5
        )
        # train set
        cols = ["class", "int_field", "int_field"]
        data = [
            [1, 1, 1], #0
            [2, 2, 2], #1 - dist from c1 = e^(sqrt(1+1)/(-2*0.5)) = 0.2431167344342142
            [3, 3, 3], #2 - dist from c1 = e^(sqrt(4+4)/(-2*0.5)) = 0.5679707120121922
            [1, 1, 1], #0
            [2, 2, 2], #1
            [3, 3, 3], #2
            [1, 1, 1], #0
        ]
        train_df = pd.DataFrame(data, columns=cols)
        # test set
        cols = ["class", "int_field", "int_field"]
        data = [
            ["c1", 1, 1], #0
        ]
        test_df = pd.DataFrame(data, columns=cols)
        neighbors = knnp._find_datapoint_k_neighbors(0, "class", train_df, test_df)
        self.assertEqual(len(neighbors), 5)
        self.assertEqual([0.059105746561956225, 0.2431167344342142, 1.0, 1.0, 1.0], sorted([neighbor[0] for neighbor in neighbors]))
        self.assertEqual([1, 1, 1, 2, 3], sorted([neighbor[1] for neighbor in neighbors]))
        # Classification by majority neghbor label
        prediction = knnp._classify_point_from_neighbors(neighbors)
        # answer should be...
        # (0.059105746561956225*3 + 0.2431167344342142*2 + 1.0*1 + 1.0*1 + 1.0*1)
        # / (0.059105746561956225 + 0.2431167344342142 + 1.0 + 1.0 + 1.0)
        # = 1.1094197104033785
        self.assertEqual(prediction, 1.1094197104033785)

    # Demonstrate an example being edited out of the training set using edited nearest neighbor.
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
        edited_train_df = knnp.make_edited_k_nn_train_set(
            "class",
            train_df,
            remove_correct=False # This means misclassified points are removed
        )
        # the last three items in the data frame should be removed as they will be missclassified
        self.assertEqual(6, len(edited_train_df))
        self.assertEqual(['c1', 'c1', 'c2', 'c2', 'c3', 'c3'], sorted(edited_train_df["class"].to_list()))

    # Demonstrate an example being added to the training set using condensed nearest neighbor.
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
            ["c1", 1, 1], #0 - first copy of c1 will have been added, this wont be missclassified and not added
            ["c2", 2, 2], #1 - first copy of c2 will have been added, this wont be missclassified and not added
            ["c3", 3, 3], #2 - first copy of c3 will have been added, this wont be missclassified and not added
        ]
        train_df = pd.DataFrame(data, columns=cols)
        df_indicies = knnp.make_condensed_k_nn_train_set(
            "class",
            train_df,
        )
        # only one item from each class should end up in the train set...
        self.assertEqual(3, len(df_indicies))
        self.assertEqual(['c1', 'c2', 'c3'], sorted(train_df.loc[df_indicies]["class"].to_list()))

    def test_k_nearest_neighbor_classification(self):
        knnp = KNearestNeighborPredictor(
            3, False
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
        pred_df = knnp.condensed_k_nearest_neighbor(
            "class",
            train_df,
            test_df
        )
        print(pred_df)
        self.assertTrue(pd.Series([1.0, 2.0, 3.0]).equals(pred_df["prediction"]))