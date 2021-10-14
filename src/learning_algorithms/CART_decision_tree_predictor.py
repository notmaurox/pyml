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

class Node(object):
     
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

    def mean_label(self):
        if self.classification is None:
            self.classification = self.data[self.class_col].mean()
        return self.classification

    def can_classify(self):
        if self.lte_feature_child is None and self.gt_feature_child is None:
            return True
        return False


class RegressionTree(object):

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
        gt_avrg = gt_sum / len(gt_classes)
        for index in range(len(gt_classes)):
            gt_classes[index] = (gt_classes[index] - gt_avrg) ** 2
        gt_mse = gt_classes.mean()
        ## Calculate mse of classes with feature value less then or equal to feature mean
        lte_avrg = lte_sum / len(lte_classes)
        for index in range(len(lte_classes)):
            lte_classes[index] = (lte_classes[index] - lte_avrg) ** 2
        print(lte_classes)
        lte_mse = lte_classes.mean()
        ## Return avrg MSE across both classes
        return {
           "avrg_mse": (gt_mse + lte_mse) / 2,
           "feature_split_val" : feature_mean,
           "lte_feature_split_indicies": lte_row_indicies,
           "lte_examples_mse": lte_mse,
           "gt_feature_split_indicies": gt_row_indicies,
           "gt_examples_mse": gt_mse
        }

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
        return (
            best_feature, feature_split_val[best_feature],
            lte_indicies[best_feature], gte_indicies[best_feature],
            lte_mse[best_feature], gt_mse[best_feature]
        )

    def __init__(self, data: pd.DataFrame, class_col: str, partitiaion_mse_threshold: float):
        self.node_count = 1
        self.class_col = class_col
        self.mse_threshold = partitiaion_mse_threshold
        self.root = Node(self.node_count, data, class_col, float("inf"))
        self.node_store = {self.root.id: self.root}

    def build_tree(self, node: Node):
        if node.split_mse <= self.mse_threshold:
            return
        best_feature, feature_split, lte_indicies, gte_indicies, lte_mse, gt_mse = RegressionTree.pick_best_feature_to_split(
            node.data, node.class_col)
        node.feature = best_feature
        node.feature_split_val = feature_split
        # Make lte child...
        self.node_count += 1
        lte_child_node = Node(
            id=self.node_count,
            data=node.data.loc[lte_indicies],
            class_col=node.class_col,
            initial_mse=lte_mse

        )
        node.lte_feature_child = lte_child_node
        self.build_tree(node.lte_feature_child)
        # Make gt child...
        self.node_count += 1
        gt_child_node = Node(
            id=self.node_count,
            data=node.data.loc[gte_indicies],
            class_col=node.class_col,
            initial_mse=gt_mse

        )
        node.gt_feature_child = gt_child_node
        self.build_tree(node.gt_feature_child)

    def classify_example(self, example: pd.DataFrame):
        return self._traverse_tree(self.root, example)
    
    def _traverse_tree(self, node: Node, example: pd.DataFrame):
        # If traversal reaches a leaf, stop traversing
        if node.can_classify():
            return node.mean_label()
        # Continue traversal...
        if example[node.feature] <= node.feature_split_val:
            return self._traverse_tree(node.lte_feature_child, example)
        elif example[node.feature] > node.feature_split_val:
            return self._traverse_tree(node.gt_feature_child, example)