import logging
import pandas as pd
import numpy as np

from typing import Dict, List, Tuple
from math import ceil


LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

class DataTransformer(object):

    @staticmethod
    def identify_cols_with_missing_data(df: pd.DataFrame) -> List[str]:
        cols_with_na = df.columns[df.isna().any()].tolist()
        return cols_with_na

    # 1.2
    @staticmethod
    def impute_missing_vales_with_mean(df: pd.DataFrame, col: str) -> pd.DataFrame:
        if col not in df.columns:
            raise ValueError(f'Column missing from dataframe: {col}')
        col_mean=df[col].mean()
        df[col].fillna(value=col_mean, inplace=True)
        return df

    # 1.3 - Ordinal Data
    # Provide a map where keys are values in the columns and values are integers those keys should be replaced with
    @staticmethod
    def handle_ordinal_col_with_map(df: pd.DataFrame, col: str, transform_map: Dict[str, int]) -> pd.DataFrame:
        if col not in df.columns:
            raise ValueError(f'Column missing from dataframe: {col}')
        LOG.info(f"Modifying ordinal col {col} with map")
        df[col] = df[col].map(transform_map, na_action=None)
        return df

    # 1.3 - Nominal Data
    # Break out a column with N possible string values into N binary features with names [ORIG_COL]_[VALUE]
    @staticmethod
    def handle_nomal_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
        if col not in df.columns:
            raise ValueError(f'Column missing from dataframe: {col}')
        LOG.info(f"Modifying nominal col {col}")
        df = pd.get_dummies(df, columns=[col], prefix=col)
        return df

    # 1.4 - Discretization
    # Take values within certain ranges and replace them with a single value
    @staticmethod
    def discretize_col(df: pd.DataFrame, col: str, num_buckets: int, equal_width: bool=True, equal_freq: bool=False) -> pd.DataFrame:
        if equal_width and equal_freq:
            raise ValueError("discretize can have equal width or equal freq buckets but not both")
        if col not in df.columns:
            raise ValueError(f'Column missing from dataframe: {col}')
        if equal_freq:
            df[col] = pd.qcut(df[col], q=num_buckets, duplicates='drop')
            if len(df[col].value_counts()) != num_buckets:
                raise ValueError(f"qcut unable to split column {col} into {str(num_buckets)} equal frequency buckets")
        else:
            df[col] = pd.cut(df[col], bins=num_buckets)
            if len(df[col].value_counts()) != num_buckets:
                raise ValueError(f"cut unable to split column {col} into {str(num_buckets)} equal frequency buckets")
        return df

    # 1.5 - Standardization
    # Apply z-score standardization to a column in a dataframe. If an optional test_df is provided, the mean and std dev
    # from the first data frame is used to normalize the same column in test_df. Turns feature into score that represents
    # how mant standard deviations the value is from the mean
    @staticmethod
    def z_score_standardize(df: pd.DataFrame, col: str, test_df: pd.DataFrame=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if col not in df.columns:
            raise ValueError(f'Column missing from dataframe: {col}')
        LOG.info(f"Normalizing data frame by z score standardization on col {col}")
        col_mean = df[col].mean()
        col_std_dev = df[col].std(ddof=0)
        print(col_mean, col_std_dev)
        df[col] = (df[col] - col_mean)/col_std_dev
        if not test_df is None:
            if col not in test_df.columns:
                raise ValueError(f'Column missing from test dataframe: {col}')
            LOG.info(f"Normalizing test set data frame by z score standardization on col {col}")
            test_df[col] = (test_df[col] - col_mean)/col_std_dev
        return df, test_df

    # 1.6 - Cross validation
    # train set (k-1/k), test set(1/k), 
    # hyperparameter set (proportion of data set aside pre splitting to k slices)
    # hyperparmeters cannot be learned from the model, set aside N% of data to test diff hyperparameters
    # Returns two objects, the first is a list of k slices where each item is a list of indicies in the dataframe
    # of the rows in that slice. The second is a list of incidies in the dataframe of rows that belong to the 
    # hyperparmeter validation set. In order to populate the validation set, pass an optional "True" as the 4th parameter
    @staticmethod
    def define_k_folds(
        df: pd.DataFrame, k: int, class_col: str, sort_values:bool=False
    ) -> List[List[int]]:
        fold_indicies = [[] for _ in range(k)]
        label_counts = df[class_col].value_counts()
        class_cnt_tuples = tuple(zip(
            [label for label in label_counts.index],
            label_counts.values
        ))
        for label, label_count in class_cnt_tuples:
            label_rows = df[df[class_col] == label]
            slices = np.array_split(label_rows, k)
            for slice_index in range(len(slices)):
                slice_item_indicies = slices[slice_index].index.values.tolist()
                fold_indicies[slice_index] = fold_indicies[slice_index] + slice_item_indicies
        return fold_indicies

    @staticmethod
    def produce_k_fold_cross_validation_sets(
        df: pd.DataFrame, k: int, class_col: str, make_hyperparam_set:bool=False, hyperparam_set_proportion:float=0.20
    ) -> Tuple[List[Tuple[List[int], List[int]]], List[int]]:
        # isolate dataframe indicies for hyperparam validation set and use the remaining indicies to generate k slices
        if make_hyperparam_set:
            num_slices = int(len(df) * (1 // hyperparam_set_proportion))
            print(num_slices)
            k_fold_slices = DataTransformer.define_k_folds(df, num_slices, class_col)
            hyperparam_indicies = k_fold_slices[0]
            # make k-fold cross-validation set excluding indicies used in hyperparam set
            hyperparam_indicies_df = df.index.isin(hyperparam_indicies)
            k_fold_slices = DataTransformer.define_k_folds(df[~hyperparam_indicies_df], k, class_col)
        else:
            # without the make_hyperparam_set flag, an empty list of hyperparam validation indicies set will be returned
            hyperparam_indicies = []
            # make k-fold cross-validation using full input dataframe
            k_fold_slices = DataTransformer.define_k_folds(df, k, class_col)
        # make k-fold cross-validation set
        k_fold_test_train_sets = []
        for test_fold_index in range(len(k_fold_slices)):
            train_set_indicies = [index for index in range(len(k_fold_slices)) if index != test_fold_index]
            train_set = []
            for train_set_index in train_set_indicies:
                train_set = train_set + k_fold_slices[train_set_index]
            test_set = k_fold_slices[test_fold_index]
            k_fold_test_train_sets.append( (train_set, test_set) )
        return k_fold_test_train_sets, hyperparam_indicies

    # Apply log transform to column of dataframe
    @staticmethod
    def log_transform_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
        if col not in df.columns:
            raise ValueError(f'Column missing from dataframe: {col}')
        df[col] = np.log10(df[col])
        return df

