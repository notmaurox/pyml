import logging
import sys
import os, pathlib
import pandas as pd
import numpy as np

from typing import List
from statistics import mode

import os, pathlib
PATH_TO_SRC_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), "../", "src/")
sys.path.insert(0, PATH_TO_SRC_DIR)

from data_utils.metrics_evaluator import MetricsEvaluator

# Logging stuff
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)
# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# LOG.addHandler(handler)

PRED_COL_NAME = "prediction"

# TODO logging
class KNearestNeighborPredictor(object):

    # TODO comment
    def __init__(self, k:int, do_regression:bool, allowed_error:float=None):
        self.k = k
        self.bandwidth_param = 0.00
        self.allowed_error = allowed_error
        self.training_set_sizes = []
        self.classification_scores = []
        self.do_regression = do_regression

    # TODO comment
    def k_nearest_neighbor(self, class_col: str, train_set: pd.DataFrame, test_set: pd.DataFrame) -> pd.DataFrame:
        LOG.info(f"Running K nearest neighbor prediction by majority label on DataFrame column {class_col}")
        if class_col not in train_set.columns:
            raise ValueError(f'Column missing from dataframe: {class_col}')
        # Run prediction
        predicted_classes = []
        for test_row_index in test_set.index:
            neighbors = []
            for train_row_index in train_set.index:
                # Calculate Frobenius norm aka euclidean norm between two vectors
                euclidean_dist = np.linalg.norm(
                    test_set.loc[test_row_index].drop([class_col])
                    - train_set.loc[train_row_index].drop([class_col])
                )
                # Lazy method for tracking distance of K NN and associated label. Could be improved with min heap
                if len(neighbors) < self.k: 
                    neighbors.append([euclidean_dist, train_set.loc[train_row_index][class_col]])
                else:
                    for neighbor_index in range(len(neighbors)):
                        if neighbors[neighbor_index][0] > euclidean_dist:
                            neighbors[neighbor_index][0] = euclidean_dist
                            neighbors[neighbor_index][1] = train_set.loc[train_row_index][class_col]
                            break
            predicted_classes.append(mode([neighbor[1] for neighbor in neighbors]))
        # Need to use indicies of test set when adding prediction series to test set df
        test_set[PRED_COL_NAME] = pd.Series(predicted_classes, index=test_set.index)
        return test_set

    # TODO comment
    # Iterate through the training set and classify using the other points in the training set.
    # If rmv_correct==False, then examples that are misclasified are removed from the set. 
    # If rmv_correct==True, then examples that are correctly classified are removed from the set. 
    def make_edited_k_nn_train_set(
        self, class_col: str, train_set: pd.DataFrame, remove_correct: bool=False) -> List[int]:
        LOG.info(f"Generating indicies for edited k nearest neighbot train set on DataFrame column {class_col}")
        if class_col not in train_set.columns:
            raise ValueError(f'Column missing from dataframe: {class_col}')
        for train_set_index in train_set.index:
            test_dp_df = train_set.loc[train_set_index].copy().to_frame().transpose()
            train_set_wo_test = train_set[~train_set.index.isin([train_set_index])]
            if len(train_set_wo_test.index) == 0:
                return train_set_wo_test
            single_train_set_pred = self.k_nearest_neighbor(
                class_col, train_set_wo_test, test_dp_df
            )
            # Check if missclassified
            was_missclassified = False
            if self.allowed_error:
                if (abs(single_train_set_pred.loc[train_set_index][PRED_COL_NAME]-single_train_set_pred.loc[train_set_index][class_col])
                    > self.allowed_error):
                    was_missclassified = True
            else:
                if single_train_set_pred.loc[train_set_index][PRED_COL_NAME] != single_train_set_pred.loc[train_set_index][class_col]:
                    was_missclassified = True
            # if removing miss classified and data point was misclassified
            if not remove_correct and was_missclassified:
                train_set = train_set.drop(train_set_index)
            # if removing correctly classified and point was not misclassified
            elif remove_correct and not was_missclassified:
                train_set = train_set.drop(train_set_index)
        # At the end of for loop, edited_train_set_indicies will contain list of incorrectly classified datapoints
        return train_set
        
    #TODO comment
    # Repeat untill performance on validation set no longer improves or no further points are removed.
    def edited_k_nearest_neighbor(
        self, class_col: str, train_set: pd.DataFrame, test_set: pd.DataFrame, remove_correct: bool=False) -> pd.DataFrame:
        LOG.info(f"Running K nearest neighbor prediction by majority label on DataFrame column {class_col}")
        if class_col not in train_set.columns:
            raise ValueError(f'Column missing from dataframe: {class_col}')
        # Run prediction
        edited_training_set = train_set.copy()
        self.training_set_sizes = [len(edited_training_set)]
        self.classification_scores = [0.00]
        best_classification = None
        while True:
            edited_training_set = self.make_edited_k_nn_train_set(
                class_col, edited_training_set, remove_correct
            )
            print(len(edited_training_set))
            predicted_test_set = self.k_nearest_neighbor(
                class_col, edited_training_set, test_set
            )
            classification_score = MetricsEvaluator.calculate_classification_score(
                predicted_test_set[class_col], predicted_test_set[PRED_COL_NAME]
            )
            print(classification_score)
            # when an iteration fails to decrease training set size or improve classification score,
            if (len(edited_training_set) >= self.training_set_sizes[-1]
                or classification_score < self.classification_scores[-1]):
                return best_classification
            else:
                self.training_set_sizes.append(len(edited_training_set))
                self.classification_scores.append(classification_score)
                best_classification = predicted_test_set.copy()

    # TODO comment
    # continue to iterate through training set in random order untill a full loop through the training set does not
    # add any more points to the set of edited train set indicies
    def make_condensed_k_nn_train_set(self, class_col: str, train_set: pd.DataFrame) -> List[int]:
        LOG.info(f"Generating indicies for edited k nearest neighbot train set on DataFrame column {class_col}")
        if class_col not in train_set.columns:
            raise ValueError(f'Column missing from dataframe: {class_col}')
        # Build train set...
        edited_train_set_indicies = []
        while True:
            edited_train_set_size_pre_train_set_iteration = len(edited_train_set_indicies)
            train_set = train_set.sample(frac=1) # Shuffle train set order
            for train_set_index in train_set.index:
                # Base case, set of edited train set indicies is empty...
                if edited_train_set_indicies == []:
                    edited_train_set_indicies.append(train_set_index)
                    continue
                closest_point_dist = float('inf')
                closest_point_index = None
                for edited_train_set_index in edited_train_set_indicies:
                    euclidean_dist = np.linalg.norm(
                        train_set.loc[train_set_index].drop([class_col])
                        - train_set.loc[edited_train_set_index].drop([class_col])
                    )
                    if euclidean_dist < closest_point_dist:
                        closest_point_dist = euclidean_dist
                        closest_point_index = edited_train_set_index
                print(train_set_index, closest_point_index)
                if self.allowed_error:
                    if (abs(train_set.loc[train_set_index][class_col]-train_set.loc[closest_point_index][class_col])
                        > self.allowed_error):
                        edited_train_set_indicies.append(train_set_index)
                else:
                    if train_set.loc[train_set_index][class_col] != train_set.loc[closest_point_index][class_col]:
                        edited_train_set_indicies.append(train_set_index)
            if len(edited_train_set_indicies) != edited_train_set_size_pre_train_set_iteration:
                return edited_train_set_indicies

    #TODO comment
    def condensed_k_nearest_neighbor(self, class_col: str, train_set: pd.DataFrame, test_set: pd.DataFrame) -> pd.DataFrame:
        LOG.info(f"Running K nearest neighbor prediction by majority label on DataFrame column {class_col}")
        if class_col not in train_set.columns:
            raise ValueError(f'Column missing from dataframe: {class_col}')
        # Make condensed training set
        edited_training_set_indicies = self.make_condensed_k_nn_train_set(
            class_col, train_set
        )
        # Run using condensed training set
        predicted_test_set = self.k_nearest_neighbor(
            class_col, train_set.iloc[edited_training_set_indicies], test_set
        )
        return predicted_test_set

