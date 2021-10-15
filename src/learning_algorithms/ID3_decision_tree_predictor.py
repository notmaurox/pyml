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
     
    def __init__(self, id: int, data: pd.DataFrame, class_col: str):
        self.id = id
        self.data = data
        self.class_col = class_col
        self.feature = None
        self.classification = None
        self.children = {} # Key = feature value, Value = child node with examples where feature = value

    def majority_label(self):
        if self.classification is None:
            self.classification = self.data[self.class_col].value_counts().idxmax()
        return self.classification

    def can_classify(self):
        if len(self.children) == 0:
            return True
        return False

    def is_pure(self):
        class_labels = self.data[self.class_col].unique()
        if len(class_labels) == 1:
            return True
        return False

class ID3ClassificationTree(object):

    @staticmethod
    def handle_numeric_attributes(data: pd.DataFrame, class_col: str):
        for col in data.columns:
            if np.issubdtype(data[col].dtype, np.number):
                print(col)
                data_mod = data.copy()
                # Get the average for each class...
                data_mod.sort_values(col, axis=0)
                class_means = []
                for class_name in data[class_col].unique():
                    print(data.loc[data[class_col] == class_name])
                    class_mean = data.loc[data[class_col] == class_name][col].mean()
                    class_means.append((class_name, class_mean))
                class_means.sort(key=lambda x: x[1])
                print(class_means)
                for i in range(len(class_means)-1):
                    mid_pt_between_classes = (class_means[i][1] + class_means[i+1][1]) / 2
                    indicies_to_replace = data_mod.loc[data_mod[col] <= mid_pt_between_classes].index.to_list()
                    data_mod = data_mod.drop(indicies_to_replace, axis=0)
                    data.loc[indicies_to_replace, col] = f"{class_means[i][0]}_feat_val_bucket"
                # Replace anything larger than last midpoint between classes...
                indicies_to_replace = data_mod.loc[data_mod[col] > mid_pt_between_classes].index.to_list()
                data_mod.drop(indicies_to_replace, axis=0)
                data.loc[indicies_to_replace, col] = f"{class_means[-1][0]}_feat_val_bucket"
        return data


    @staticmethod
    def calc_entropy(class_breakdown: Dict[str, int], total_examples: int) -> float:
        entropy = 0.00
        for _, class_count in class_breakdown.items():
            entropy -= (class_count / total_examples) * math.log((class_count / total_examples), 2)
        return entropy

    @staticmethod
    def calculate_feature_gain_and_info_val(feature: pd.Series, labels: pd.Series) -> Tuple[float, float]:
        feature_vals_indexed = defaultdict(lambda: defaultdict(int))
        class_count = defaultdict(int)
        total_examples = 0
        for row_index in feature.index:
            feature_val_key = feature.loc[row_index]
            class_key = labels.loc[row_index]
            # Update feature val representation counter
            feature_val_counts = feature_vals_indexed.get(feature_val_key, defaultdict(int))
            feature_val_counts[class_key] += 1
            feature_vals_indexed[feature_val_key] = feature_val_counts
            # Update class representation counter
            class_count[class_key] += 1
            total_examples += 1
        # Calculate feature entropy and information value
        info_val, feature_entropy = 0.00, 0.00
        for feature_val_key, feature_val_counts in feature_vals_indexed.items():
            feature_val_total_count = 0
            for class_key, feature_val_class_count in feature_val_counts.items():
                feature_val_total_count += feature_val_class_count
            feature_val_probability = ( feature_val_total_count / total_examples )
            info_val += (feature_val_probability * math.log(feature_val_probability, 2))
            feature_val_entropy = feature_val_probability * ID3ClassificationTree.calc_entropy(feature_val_counts, feature_val_total_count)
            feature_entropy += feature_val_entropy
        info_val = info_val * (-1)
        # print(ID3ClassificationTree.calc_entropy(class_count, total_examples), feature_entropy)
        gain = ID3ClassificationTree.calc_entropy(class_count, total_examples) - feature_entropy
        return feature_entropy, gain, info_val

    @staticmethod
    def pick_best_feature_to_split(data: pd.DataFrame, class_col: str) -> str:
        feature_scores = {}
        for feature in data.columns:
            if feature == class_col:
                continue
            _, f_gain, f_info_val = ID3ClassificationTree.calculate_feature_gain_and_info_val(
                feature=data[feature],
                labels=data[class_col]
            )
            if f_gain == 0:
                continue
            feature_scores[feature] = (f_gain / f_info_val)
        return max(feature_scores, key=feature_scores.get)

    def __init__(self, data: pd.DataFrame, class_col: str):
        self.node_count = 1
        self.class_col = class_col
        self.root = Node(self.node_count, data, class_col)
        self.node_store = {self.root.id: self.root}

    def build_tree(self, node: Node):
        if node.is_pure():
            return
        node.feature = ID3ClassificationTree.pick_best_feature_to_split(node.data, node.class_col)
        for feature_val in node.data[node.feature].unique():
            # Select elements for children...
            self.node_count += 1
            child_node = Node(
                id=self.node_count,
                data=node.data.loc[node.data[node.feature] == feature_val],
                class_col=node.class_col
            )
            node.children[feature_val] = child_node
            self.node_store[child_node.id] = child_node
        for _, child_node in node.children.items():
            self.build_tree(child_node)

    def _traverse_tree(self, node: Node, example: pd.DataFrame):
        print(node.id)
        # If traversal reaches a leaf, stop traversing
        if node.can_classify():
            return node.majority_label()
        # If attempting to follow a branch that doesn't exist, stop traversing
        if node.children.get(example[node.feature]) is None:
            return node.majority_label()
        # Continue traversal...
        return self._traverse_tree(node.children[example[node.feature]], example)

    def classify_example(self, example: pd.DataFrame):
        return self._traverse_tree(self.root, example)

    def calculate_precision_on_set(self, data_set: pd.DataFrame,) -> float:
        correct_classifications = 0
        for row_index in data_set.index:
            prediction = self.classify_example(data_set.loc[row_index])
            if prediction == data_set.loc[row_index][self.class_col]:
                correct_classifications += 1
        return correct_classifications / len(data_set)

    def prune_tree(self, validation_set: pd.DataFrame):
        # Calculate 
        best_prediction_accuracy = self.calculate_precision_on_set(validation_set)
        print(best_prediction_accuracy)
        # iterate through nodes in reverse order...
        key_list = list(self.node_store.keys())
        key_list.reverse()
        print(key_list)
        for node_id in key_list:
            node = self.node_store[node_id]
            # Skip leaf nodes as there are no clidren of these nodes to prune from the tree
            if node.children == {}:
                continue
            previous_children = copy.deepcopy(node.children)
            node.children = {}
            prediction_accuracy_wo_node = self.calculate_precision_on_set(validation_set)
            print(prediction_accuracy_wo_node)
            # If we do better with node having no children, remove it's children...
            if prediction_accuracy_wo_node >= best_prediction_accuracy:
                node.children = {}
                for _, child_node in previous_children.items():
                    self.node_store.pop(child_node.id)
                best_prediction_accuracy = prediction_accuracy_wo_node
            else:
                node.children = previous_children




        
    




