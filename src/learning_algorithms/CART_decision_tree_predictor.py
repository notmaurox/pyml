import math
import copy
import logging
import sys

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple

# Logging stuff
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
LOG.addHandler(handler)

# Node class used to build tree, in CART regression it has a branch to a child containing data points with feature value
# less than or equal to the split. The other branch contains data points with feature value greater than split.
# These are initialized with an id for indexing, a dataframe containing the points considered at that node, a class col,
# and an initial MSE for the data used to initialize the node. This way, upon initialization, a node does not need to 
# recalculate it's initial mse before deciciding to break recursive tree building.
class Node(object):
    
    # Take an id for indexing, dataframe for the points considered at that node, a class column, and initial mse
    def __init__(self, id: int, data: pd.DataFrame, class_col: str, initial_mse: float):
        self.id = id
        self.data = data
        self.class_col = class_col
        self.feature = None
        self.feature_split_val = None
        self.classification = None
        self.split_mse = initial_mse
        self.lte_feature_child = None
        self.gt_feature_child = None

    # Used to classify points by returning the mean class label of the points stored at this node.
    # Mean is only calculated once and saved for subsequent classifications
    def mean_label(self):
        if self.classification is None:
            self.classification = self.data[self.class_col].mean()
        return self.classification

    # checks that the node has no children, if it does, then 
    def can_classify(self):
        if self.lte_feature_child is None and self.gt_feature_child is None:
            return True
        return False

# Contains static methods for doing calculations that should not depend on state of the tree and consuctor + methods
# for bulding the tree.
class RegressionTree(object):

    # Takes two columns of a data frame as pandas series, one for a feature and one for a label. This does a lot
    # of indexing to save time on calculations. The feature values are only iterated once and various counters are 
    # maintained to track data points greater than the mean, less than or equal to the mean, and the mse of both.
    # These are returned as a dictionary so that specific keys can access different results of calculations...ß
    @staticmethod
    def calculate_feature_mse(feature: pd.Series, labels: pd.Series):
        feature_mean = feature.mean()
        # Use gt to track class labels where feature val is greater than feature_mean
        # Use lte to track class labels where feature val is less than or equal to feature_mean
        gt_classes, lte_classes = np.array([]), np.array([])
        gt_row_indicies, lte_row_indicies = [], []
        gt_sum, lte_sum = 0, 0
        for row_index in feature.index:
            feature_val =  feature.loc[row_index]
            class_val = labels.loc[row_index]
            if feature_val > feature_mean:
                gt_row_indicies.append(row_index)
                gt_classes = np.append(gt_classes, class_val)
                gt_sum += class_val
            elif feature_val <= feature_mean:
                lte_row_indicies.append(row_index)
                lte_classes = np.append(lte_classes, class_val)
                lte_sum += class_val
        ## Calculate mse of classes with feature value greater than feature_mean
        if len(gt_classes) > 0:
            gt_avrg = gt_sum / len(gt_classes)
            for index in range(len(gt_classes)):
                gt_classes[index] = (gt_classes[index] - gt_avrg) ** 2
            gt_mse = gt_classes.mean()
        else:
            gt_mse = 0
        ## Calculate mse of classes with feature value less then or equal to feature mean
        if len(lte_classes) > 0:
            lte_avrg = lte_sum / len(lte_classes)
            for index in range(len(lte_classes)):
                lte_classes[index] = (lte_classes[index] - lte_avrg) ** 2
            lte_mse = lte_classes.mean()
        else:
            lte_mse = 0
        ## Return avrg MSE across both classes
        to_return = {
           "avrg_mse": (gt_mse + lte_mse) / 2,
           "feature_split_val" : feature_mean,
           "lte_feature_split_indicies": lte_row_indicies,
           "lte_examples_mse": lte_mse,
           "gt_feature_split_indicies": gt_row_indicies,
           "gt_examples_mse": gt_mse
        }
        # If the feature doesn't split the data at all, it should not be considered...
        if len(lte_classes) == 0 or len(gt_classes) == 0:
            to_return["avrg_mse"] = float("inf")
        return to_return

    # Takes a pandas dataframe and a string specifying the class column. It then splits the data in each column by the 
    # column mean and uses calculate_feature_mse to get the average mse, the examples less than or equal to the split
    # and their mse, and the examples greater than or equal to the split and their mse.
    @staticmethod
    def pick_best_feature_to_split(data: pd.DataFrame, class_col: str):
        feature_mse_avrg, feature_split_val = {}, {}
        lte_indicies, gte_indicies = {}, {}
        lte_mse, gt_mse = {}, {}
        for feature in data.columns:
            if feature == class_col:
                continue
            # Cringe way of passing data from helper function
            info_dict = RegressionTree.calculate_feature_mse(
                data[feature], data[class_col])
            feature_mse_avrg[feature] = info_dict["avrg_mse"]
            feature_split_val[feature] = info_dict["feature_split_val"]
            lte_indicies[feature] = info_dict["lte_feature_split_indicies"]
            lte_mse[feature] = info_dict["lte_examples_mse"]
            gte_indicies[feature] = info_dict["gt_feature_split_indicies"]
            gt_mse[feature] = info_dict["gt_examples_mse"]
        best_feature = min(feature_mse_avrg, key=feature_mse_avrg.get)
        # Returning the indicies for examples in each parition and the mse of the examples in both, these do not need 
        # to be recalculated later. A single pass through the dataframe results in all the information required to 
        # initialize child nodes and determine if they will be leafs or not.
        return (
            best_feature, feature_split_val[best_feature],
            lte_indicies[best_feature], gte_indicies[best_feature],
            lte_mse[best_feature], gt_mse[best_feature]
        )

    # Constructor class for the CART regression tree, takes a dataframe of training examples, a string specifying the
    # class column, and a partition mse threshold. A root node is intiailized containing the entire input data set. 
    def __init__(self, data: pd.DataFrame, class_col: str, partitiaion_mse_threshold: float):
        LOG.info(f"Initializing new regression tree with allowed partition max mse of {partitiaion_mse_threshold}")
        self.node_count = 1
        self.class_col = class_col
        self.mse_threshold = partitiaion_mse_threshold
        self.root = Node(self.node_count, data, class_col, float("inf"))
        # Used to map node id's to nodes for easier access...
        self.node_store = {self.root.id: self.root}

    # Recursive build tree function that takes a node and checks that node's MSE to see if consturction should be stopped. 
    # also checks that a node does not contain two examples with identical feature values but different labels. These
    # will result in infinite recursion if the mse between the two exceeds the threshold. 
    def _build_tree(self, node: Node):
        if node.split_mse <= self.mse_threshold:
            LOG.debug(f"EARLY STOPPING - Node with id {node.id} has mse {node.split_mse} less than threshold")
            return
        # In case that training data includes two examples that are identical in feature values but different in class
        if node.data.groupby([col for col in node.data.columns if col != node.class_col]).ngroups == 1:
            return
        # Call pick_best_feature_to_split and get the best feature and all the info required to initialize children
        best_feature, feature_split, lte_indicies, gte_indicies, lte_mse, gt_mse = RegressionTree.pick_best_feature_to_split(
            node.data, node.class_col)
        node.feature = best_feature
        node.feature_split_val = feature_split
        # Make lte child containing examples with feature value less than or equal to the feature split value
        self.node_count += 1
        lte_child_node = Node(
            id=self.node_count,
            data=node.data.loc[lte_indicies],
            class_col=node.class_col,
            initial_mse=lte_mse
        )
        self.node_store[lte_child_node.id] = lte_child_node
        node.lte_feature_child = lte_child_node
        self._build_tree(node.lte_feature_child) # continue recursion on the left child....
        # Make gt child containing examples with feature value greater than the feature split value
        self.node_count += 1
        gt_child_node = Node(
            id=self.node_count,
            data=node.data.loc[gte_indicies],
            class_col=node.class_col,
            initial_mse=gt_mse

        )
        self.node_store[gt_child_node.id] = gt_child_node
        node.gt_feature_child = gt_child_node
        self._build_tree(node.gt_feature_child) # continue recursion on the right child....
    
    # Initiates recrusive build tree function on the root node..
    def build_tree(self):
        LOG.info("Building tree...")
        self._build_tree(self.root)

    # Initiates recursive classification traversal of an example data point...
    def classify_example(self, example: pd.DataFrame):
        LOG.debug(f"Classifying example: {example}")
        return self._traverse_tree(self.root, example)
    
    # Recursive tree traversal used to classify an example. 
    def _traverse_tree(self, node: Node, example: pd.DataFrame):
        # If traversal reaches a leaf, stop traversing
        if node.can_classify():
            LOG.debug(f"Reached node {node.id} with no children, returning label {node.mean_label()}")
            return node.mean_label()
        # Continue traversal...
        LOG.debug(f"At node splitting on feature {node.feature}")
        # If example has a feature value less than or equal to the split point, take the less than or equal to child
        if example[node.feature] <= node.feature_split_val:
            LOG.debug(f"Taking less than or equal to branch from split {node.feature_split_val} with feature val {example[node.feature]}")
            return self._traverse_tree(node.lte_feature_child, example)
        # If examples has a feature value greater than the split point, take the greater than child. 
        elif example[node.feature] > node.feature_split_val:
            LOG.debug(f"Taking greater than branch from split {node.feature_split_val} with feature val {example[node.feature]}")
            return self._traverse_tree(node.gt_feature_child, example)

    # Takes a collection of data points and classifies each before adding a column called "prediction" to the input
    # dataframe containing class predictions for each data point.
    def classify_examples(self, examples: pd.DataFrame):
        LOG.info(f"Classifying {len(examples)} examples with tree...")
        predicted_classes = []
        for row_index in examples.index:
            prediction = self.classify_example(examples.loc[row_index])
            LOG.debug(f"Recieved prediction {prediction} for example with label {examples.loc[row_index][self.class_col]}")
            predicted_classes.append(prediction)
        # Need to use indicies of test set when adding prediction series to test set df
        examples["prediction"] = pd.Series(predicted_classes, index=examples.index)
        return examples