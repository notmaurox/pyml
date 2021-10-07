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
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
LOG.addHandler(handler)

PRED_COL_NAME = "prediction"

# TODO logging
# Common strategy for learning algorithms is to add a column called "prediction" to the test set dataframe with the
# predicted label.
class KNearestNeighborPredictor(object):

    # Class constructor. Takes K to define the number of neighbots to consider and a boolean. If False, classifiction
    # is done in learning, if True, regression is done in learning. Allows for configurable sigma and allowed error
    # for classification tasks. 
    def __init__(self, k:int, do_regression:bool, sigma:float=2.5, allowed_error:float=None):
        LOG.info(f"Initializing K-nn predictor with k={k} and regression={do_regression}")
        self.k = k
        self.bandwidth_param = 0.00
        self.allowed_error = allowed_error
        self.training_set_sizes = []
        self.classification_scores = []
        self.do_regression = do_regression
        self.sigma = sigma

    # Helper method that takes an index of a row in the test_set dataframe, the name of the class column, and the
    # train set. Based on configurations of the calling object, the k nearest neigbors are identified using 
    # euclidean distance or gaussian kernel. A list is returned containing a single tuple for each neighbor.
    # The first item in the tuple is the distance/gaussian kernel score and the second is the label.
    def _find_datapoint_k_neighbors(
        self, test_row_index:int, class_col: str, train_set: pd.DataFrame, test_set: pd.DataFrame):
        LOG.debug(f"Finding nearest neighbors for test row index {test_row_index}")
        neighbors = []
        for train_row_index in train_set.index:
            # Calculate Frobenius norm aka euclidean norm between two vectors
            dist = abs(np.linalg.norm(
                test_set.loc[test_row_index].drop([class_col])
                - train_set.loc[train_row_index].drop([class_col])
            ))
            if self.do_regression:
                dist = np.exp(dist / (-2 * self.sigma))
            # Lazy method for tracking distance of K NN and associated label. Could be improved with min heap
            if len(neighbors) < self.k: 
                neighbors.append([dist, train_set.loc[train_row_index][class_col]])
            else:
                for neighbor_index in range(len(neighbors)):
                    # When using gaussian kernel values closer to 1 means more similar... select largest
                    # When using euclidean distance... select smallest 
                    if ((self.do_regression and neighbors[neighbor_index][0] < dist)
                        or (not self.do_regression and neighbors[neighbor_index][0] > dist)):
                        neighbors[neighbor_index][0] = dist
                        neighbors[neighbor_index][1] = train_set.loc[train_row_index][class_col]
                        break
        return neighbors

    # neighbors is a list of tuples where each tuple is a neighbor and is returned by  _find_datapoint_k_neighbors.
    # The first item in each tuple is the distance to the query point and the second item in each tuple is the neighbor
    # label. Based on configuration of the calling object, this will either return the majority label when doing 
    # classification or the gaussian regression value when doing regression.  
    def _classify_point_from_neighbors(self, neighbors):
        LOG.debug("Classifying point from neigbhors")
        if not self.do_regression:
            return mode([neighbor[1] for neighbor in neighbors])
        else:
            denominator = sum([neighbor[0] for neighbor in neighbors])
            if denominator == 0:
                return 0
            regression_prediction = (
                sum([neighbor[0]*neighbor[1] for neighbor in neighbors])
                / sum([neighbor[0] for neighbor in neighbors])
            )
            return regression_prediction

    # Takes a string designating the class column, a datafraim of training points and a dataframe of test points.
    # test points are iterated and k nearest neighbor classification is done using the training points.
    # Based on configuration of calling object, classification or regression is done. A new column is added to the 
    # test_set df called "prediction" that has the predicted label. 
    def k_nearest_neighbor(self, class_col: str, train_set: pd.DataFrame, test_set: pd.DataFrame) -> pd.DataFrame:
        LOG.debug(f"Running K nearest neighbor prediction by majority label on DataFrame column {class_col}")
        if class_col not in train_set.columns:
            raise ValueError(f'Column missing from dataframe: {class_col}')
        # Run prediction
        predicted_classes = []
        for test_row_index in test_set.index:
            neighbors = self._find_datapoint_k_neighbors(test_row_index, class_col, train_set, test_set)
            predicted_classes.append(self._classify_point_from_neighbors(neighbors))
        # Need to use indicies of test set when adding prediction series to test set df
        test_set[PRED_COL_NAME] = pd.Series(predicted_classes, index=test_set.index)
        return test_set

    # This method generates a modified training set for edited k nn by classifying each point using all other points in
    # the dataframe via k-nn. It returns a the training set with either misclassified
    # or correctly classified rows dropped from the data frame depending on value of remove_correct param. 
    # Iterate through the training set and classify using the other points in the training set.
    # If rmv_correct==False, then examples that are misclasified are removed from the set. 
    # If rmv_correct==True, then examples that are correctly classified are removed from the set. 
    def make_edited_k_nn_train_set(
        self, class_col: str, train_set: pd.DataFrame, remove_correct: bool=False) -> pd.DataFrame:
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

    # Helper function used to determine if score has improved in edited K nn
    def _has_score_improved(self, new_score: float, old_score: float) -> bool:
        # If we are doing classification, larger scores are better as that is proportion classified correctly.
        if not self.do_regression:
            return new_score > old_score
        # If doing regression, score is MSE, smaller is better
        return new_score < old_score

    # Repeats a loop until the size of the edited training set does not change or the classification score does not repeat.
    # First generates an edited training set with make_edited_k_nn_train_set and then uses that for classification.
    # Then if doing regression, calculates classification accruacy, if doing regression calculates MSE.
    def edited_k_nearest_neighbor(
        self, class_col: str, train_set: pd.DataFrame, test_set: pd.DataFrame, remove_correct: bool=False) -> pd.DataFrame:
        LOG.info(f"Running edited K nearest neighbor prediction by majority label on DataFrame column {class_col}")
        if class_col not in train_set.columns:
            raise ValueError(f'Column missing from dataframe: {class_col}')
        # Run prediction
        edited_training_set = train_set.copy()
        self.training_set_sizes = [len(edited_training_set)]
        if self.do_regression:
            self.classification_scores = [float('inf')]
        else:
            self.classification_scores = [-1]
        best_classification = None
        while True:
            edited_training_set = self.make_edited_k_nn_train_set(
                class_col, edited_training_set, remove_correct
            )
            predicted_test_set = self.k_nearest_neighbor(
                class_col, edited_training_set, test_set
            )
            # If classification, use classification accruacy.
            if not self.do_regression:
                classification_score = MetricsEvaluator.calculate_classification_score(
                    predicted_test_set[class_col], predicted_test_set[PRED_COL_NAME]
                )
            else: # If doing regression, use MSE...
                classification_score = MetricsEvaluator.calculate_mean_squared_error(
                    predicted_test_set[class_col], predicted_test_set[PRED_COL_NAME]
                )
            print(predicted_test_set)
            print(len(edited_training_set), classification_score)
            # when an iteration fails to decrease training set size or improve classification score,
            if (self._has_score_improved(classification_score, self.classification_scores[-1])
                and len(edited_training_set) <= self.training_set_sizes[-1]):
                self.training_set_sizes.append(len(edited_training_set))
                self.classification_scores.append(classification_score)
                best_classification = predicted_test_set.copy()
            else:
                return best_classification

    # Given a training set and a class column, initialize an edited set and iterate through the training set in random
    # order. For each data point in the training set, find its closest neighbor in the edited set. If this closest
    # neigbhor does not match the class label, add the current data point to the edited set.
    # Continue to iterate through training set in random order untill a full loop through the training set does not
    # add any more points to the set of edited train set indicies. Return a list of dataframe indicies as the edited
    # set
    def make_condensed_k_nn_train_set(self, class_col: str, train_set: pd.DataFrame) -> List[int]:
        LOG.info(f"Generating indicies for condensed k nearest neighbot train set on DataFrame column {class_col}")
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
                if self.allowed_error != None:
                    if (abs(train_set.loc[train_set_index][class_col]-train_set.loc[closest_point_index][class_col])
                        > self.allowed_error):
                        edited_train_set_indicies.append(train_set_index)
                else:
                    if train_set.loc[train_set_index][class_col] != train_set.loc[closest_point_index][class_col]:
                        edited_train_set_indicies.append(train_set_index)
            if len(edited_train_set_indicies) != edited_train_set_size_pre_train_set_iteration:
                return edited_train_set_indicies

    # Given a class column, a training set dataframe, and a test set dataframe, perform condensed nearest neighbor
    # by first creating an edited train set and then using that edited set of nearest neighbor. 
    def condensed_k_nearest_neighbor(self, class_col: str, train_set: pd.DataFrame, test_set: pd.DataFrame) -> pd.DataFrame:
        LOG.info(f"Running condensed K nearest neighbor prediction by majority label on DataFrame column {class_col}")
        if class_col not in train_set.columns:
            raise ValueError(f'Column missing from dataframe: {class_col}')
        # Make condensed training set
        edited_training_set_indicies = self.make_condensed_k_nn_train_set(
            class_col, train_set
        )
        # Run using condensed training set
        predicted_test_set = self.k_nearest_neighbor(
            class_col, train_set.loc[edited_training_set_indicies], test_set
        )
        return predicted_test_set

