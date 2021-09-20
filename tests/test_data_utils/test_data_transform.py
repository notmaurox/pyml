import unittest
import sys
import os
import pathlib

import pandas as pd
import numpy as np

PATH_TO_SRC_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), "../", "../", "src/")
sys.path.insert(0, PATH_TO_SRC_DIR)

from data_utils.data_transformer import DataTransformer
from data_utils.data_loader import DataLoader

class TestDataTransformer(unittest.TestCase):

    def test_identify_cols_with_missing_data(self):
        cols = ["sample", "str_field"]
        data = [
            ["d1", "red"],
            ["d2", None],
            ["d3", "green"]
        ]
        df = pd.DataFrame(data, columns=cols)
        na_cols = DataTransformer.identify_cols_with_missing_data(df)
        self.assertEqual(na_cols, ["str_field"])

    # Imputation of missing values
    def test_impute_missing_vales_with_mean(self):
        cols = ["sample", "int_field"]
        data = [["d1", 1], ["d2", None], ["d3", 3]]
        df = pd.DataFrame(data, columns=cols)
        df = DataTransformer.impute_missing_vales_with_mean(df, "int_field")
        self.assertEqual(df["int_field"].to_list(), [1, 2, 3])

    def test_handle_ordinal_col_with_map(self):
        cols = ["sample", "str_field"]
        data = [
            ["d1", "best"],
            ["d2", "average"],
            ["d3", "worst"]
        ]
        df = pd.DataFrame(data, columns=cols)
        df = DataTransformer.handle_ordinal_col_with_map(df, "str_field",
            {
                "best": 1,
                "average": 2,
                "worst": 3
            }
        )
        self.assertEqual(df["str_field"].to_list(), [1, 2, 3])

    #mapping of non binary to categorical variable by one-hot encoding
    def test_handle_nomal_col(self):
        cols = ["sample", "str_field"]
        data = [
            ["d1", "red"],
            ["d2", "blue"],
            ["d3", "green"]
        ]
        df = pd.DataFrame(data, columns=cols)
        df = DataTransformer.handle_nomal_col(df, "str_field")
        self.assertEqual(len(df.columns), 4)

    #discritization by equal width bins
    def test_discretize_col_equal_width(self):
        cols = ["sample", "float_field"]
        data = [
            ["d1", 1.5], #0
            ["d2", 1.3], #1
            ["d3", 1.2], #2
            ["d4", 1.9], #3
            ["d5", 2.0], #4
            ["d6", 10], #5
            ["d7", 10.5], #6
            ["d8", 10.2], #7
            ["d9", 10.3], #8
            ["d10", 10.2], #9
        ]
        df = pd.DataFrame(data, columns=cols)
        df = DataTransformer.discretize_col(df, "float_field", 2, equal_width=True)
        self.assertEqual(2, len(pd.unique(df["float_field"])))

    # discretization by equal freq bins
    def test_discretize_col_equal_freq(self):
        cols = ["sample", "float_field"]
        data = [
            ["d1", 1.5], #0
            ["d2", 1.3], #1
            ["d3", 1.2], #2
            ["d4", 9], #3
            ["d5", 9], #4
            ["d6", 10], #5
            ["d7", 10.5], #6
            ["d8", 10.2], #7
            ["d9", 10.3], #8
            ["d10", 10.2], #9
        ]
        df = pd.DataFrame(data, columns=cols)
        df = DataTransformer.discretize_col(df, "float_field", 2, equal_width=False, equal_freq=True)
        self.assertEqual(2, len(pd.unique(df["float_field"])))
        self.assertEqual(df["float_field"].iloc[0], pd.Interval(1.199, 9.5, closed="right"))

    def test_z_score_standardize(self):
        cols = ["sample", "float_field"]
        data = [
            ["d1", 1.5], #0
            ["d2", 1.3], #1
            ["d3", 1.2], #2
            ["d4", 9], #3
            ["d5", 9], #4
            ["d6", 10], #5
            ["d7", 10.5], #6
            ["d8", 10.2], #7
            ["d9", 10.3], #8
            ["d10", 10.2], #9
        ]
        df = pd.DataFrame(data, columns=cols)
        df = DataTransformer.z_score_standardize(df, "float_field")
        # Just make sure this doesnt error out
        self.assertTrue(True)

    #standardize training set and apply to test set. 
    def test_z_score_standardize_test_and_train(self):
        # Make train df
        train_cols = ["sample", "float_field"]
        train_data = [
            ["d1", 5], #0
            ["d2", 5], #1
            ["d3", 5], #2
            ["d4", 5], #3
            ["d5", 5], #4
            ["d6", 4], #5
            ["d7", 4], #6
            ["d8", 4], #7
            ["d9", 4], #8
            ["d10", 4], #9
        ]
        train_df = pd.DataFrame(train_data, columns=train_cols)
        # Make test df
        test_cols = ["sample", "float_field"]
        #  training set ->>> mean = 4.5 std_dev = 0.5
        test_data = [
            ["d1", 9], #0 | (9 - -4.5)/0.5 = 9
            ["d9", -9], #8 | (-9 - -4.5)/0.5 = -27
        ]
        test_df = pd.DataFrame(test_data, columns=test_cols)
        train_df, test_df = DataTransformer.z_score_standardize(train_df, "float_field", test_df)
        # Mean 
        self.assertTrue(test_df.iloc[0]["float_field"] == 9)
        self.assertTrue(test_df.iloc[1]["float_field"] == -27) 

    def test_define_k_folds_even_folds(self):
        cols = ["sample", "str_field"]
        data = [
            ["d1", "red"], #0
            ["d2", "red"], #1
            ["d3", "red"], #2
            ["d4", "red"], #3
            ["d5", "grn"], #4
            ["d6", "grn"], #5
            ["d7", "grn"], #6
            ["d8", "grn"], #7
            ["d9", "blu"], #8
            ["d10", "blu"], #9
        ]
        df = pd.DataFrame(data, columns=cols)
        fold_indicies = DataTransformer.define_k_folds(df, 2, "str_field")
        self.assertEqual(fold_indicies[0], [0, 1, 4, 5, 8]) #2 red, 2 grn, 1 blu
        self.assertEqual(fold_indicies[1], [2, 3, 6, 7, 9]) #2 red, 2 grn, 1 blu
        # to get a dataframe from the indicies in the list...
        self.assertEqual(len(df.iloc[fold_indicies[0]]), 5)

    def test_define_k_folds_odd_folds(self):
        # Make sure that splitting into 3 folds results in folds with class distributions comparable to
        # initial dataset class distribution
        cols = ["sample", "str_field"]
        data = [
            ["d1", "red"], #0
            ["d2", "red"], #1
            ["d3", "red"], #2
            ["d4", "red"], #3
            ["d5", "grn"], #4
            ["d6", "grn"], #5
            ["d7", "grn"], #6
            ["d8", "grn"], #7
            ["d9", "blu"], #8
            ["d10", "blu"], #9
        ]
        df = pd.DataFrame(data, columns=cols)
        fold_indicies = DataTransformer.define_k_folds(df, 3, "str_field")
        self.assertEqual(fold_indicies[0], [0, 1, 4, 5, 8]) #2 red, 2 grn, 1 blu
        self.assertEqual(fold_indicies[1], [2, 6, 9]) #1 red, 1 grn, 1 blu
        self.assertEqual(fold_indicies[2], [3, 7]) # 1 grn, 1 blu
        # to get a dataframe from the indicies in the list...
        self.assertEqual(len(df.iloc[fold_indicies[0]]), 5)

    def test_produce_k_fold_cross_validation_sets(self):
        cols = ["sample", "str_field"]
        data = [
            ["d1", "red"], #0
            ["d2", "red"], #1
            ["d3", "red"], #2
            ["d4", "red"], #3
            ["d5", "grn"], #4
            ["d6", "grn"], #5
            ["d7", "grn"], #6
            ["d8", "grn"], #7
            ["d9", "blu"], #8
            ["d10", "blu"], #9
        ]
        df = pd.DataFrame(data, columns=cols)
        fold_indicies, _ = DataTransformer.produce_k_fold_cross_validation_sets(df, 3, "str_field")
        # We know from test_define_k_folds_odd_folds that these are the apropriate slices of indicies
        slice_1 = [0, 1, 4, 5, 8] #2 red, 2 grn, 1 blu
        slice_2 = [2, 6, 9] #1 red, 1 grn, 1 blu
        slice_3 = [3, 7] # 1 grn, 1 blu
        # Make sure the slices were grouped into train and test sets correctly
        self.assertEqual(fold_indicies[0], (slice_2+slice_3, slice_1)) #2 red, 2 grn, 1 blu
        self.assertEqual(fold_indicies[1], (slice_1+slice_3, slice_2)) #1 red, 1 grn, 1 blu
        self.assertEqual(fold_indicies[2], (slice_1+slice_2, slice_3)) # 1 grn, 1 blu

    def test_produce_k_fold_cross_validation_sets_weird_case(self):
        cols = ["sample", "str_field"]
        data = [
            ["d1", "red"], #0
            ["d2", "red"], #1
            ["d3", "red"], #2
            ["d4", "red"], #3
            ["d5", "grn"], #4
            ["d6", "grn"], #5
            ["d7", "grn"], #6
            ["d8", "grn"], #7
            ["d9", "blu"], #8
            ["d10", "blu"], #9
        ]
        df = pd.DataFrame(data, columns=cols)
        fold_indicies, _ = DataTransformer.produce_k_fold_cross_validation_sets(df, 3, "str_field")
        # We know from test_define_k_folds_odd_folds that these are the apropriate slices of indicies
        slice_1 = [0, 1, 4, 5, 8] #2 red, 2 grn, 1 blu
        slice_2 = [2, 6, 9] #1 red, 1 grn, 1 blu
        slice_3 = [3, 7] # 1 grn, 1 blu
        # Make sure the slices were grouped into train and test sets correctly
        self.assertEqual(fold_indicies[0], (slice_2+slice_3, slice_1)) #2 red, 2 grn, 1 blu
        self.assertEqual(fold_indicies[1], (slice_1+slice_3, slice_2)) #1 red, 1 grn, 1 blu
        self.assertEqual(fold_indicies[2], (slice_1+slice_2, slice_3)) # 1 grn, 1 blu

    def test_produce_k_fold_cross_validation_sets_with_hyperparam_set(self):
        cols = ["sample", "str_field"]
        data = [
            ["h1", "red"], # should end up in hyper param test set
            ["h2", "blu"], # should end up in hyper param test set
            ["h3", "grn"], # should end up in hyper param test set
            ["d1", "red"], #0
            ["d2", "red"], #1
            ["d3", "red"], #2
            ["d4", "red"], #3
            ["d5", "grn"], #4
            ["d6", "grn"], #5
            ["d7", "grn"], #6
            ["d8", "grn"], #7
            ["d9", "blu"], #8
            ["d10", "blu"], #9
        ]
        df = pd.DataFrame(data, columns=cols)
        fold_indicies, hyperparam_indicies = DataTransformer.produce_k_fold_cross_validation_sets(
            df, 3, "str_field",
            make_hyperparam_set=True,
            hyperparam_set_proportion=0.23
        )
        # We know from test_define_k_folds_odd_folds that these are the apropriate slices of indicies
        slice_1 = [item+3 for item in [0, 1, 4, 5, 8]] #2 red, 2 grn, 1 blu
        slice_2 = [item+3 for item in [2, 6, 9]] #1 red, 1 grn, 1 blu
        slice_3 = [item+3 for item in [3, 7]] # 1 grn, 1 blu
        # Make sure the slices were grouped into train and test sets correctly
        hyperparam_indicies.sort()
        self.assertEqual(hyperparam_indicies, [0, 1, 2])
        self.assertEqual(fold_indicies[0], (slice_2+slice_3, slice_1)) #2 red, 2 grn, 1 blu
        self.assertEqual(fold_indicies[1], (slice_1+slice_3, slice_2)) #1 red, 1 grn, 1 blu
        self.assertEqual(fold_indicies[2], (slice_1+slice_2, slice_3)) # 1 grn, 1 blu

    def test_discretize_col_equal_freq_on_forest(self):
        # Test equal width
        df = DataLoader.load_forestfires_data()
        df = DataTransformer.discretize_col(df, "area", 5, equal_width=True, equal_freq=False)
        unique_areas = df["area"].value_counts()
        print(unique_areas)
        self.assertEqual(5, len(unique_areas))
        # Test equal freq
        df = DataLoader.load_abalone_data()
        df = DataTransformer.discretize_col(df, "diameter", 5, equal_width=False, equal_freq=True)
        unique_areas = df["diameter"].value_counts()
        print(unique_areas)
        self.assertEqual(5, len(unique_areas))
        # Test equal freq broken
        df = DataLoader.load_forestfires_data()
        try:
            df = DataTransformer.discretize_col(df, "area", 5, equal_width=False, equal_freq=True)
        except ValueError:
            pass
        