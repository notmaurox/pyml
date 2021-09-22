import logging
import sys
import pandas as pd
import numpy as np

from statistics import mode

# Logging stuff
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)
# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# LOG.addHandler(handler)

class KNearestNeighborPredictor(object):

    @staticmethod
    def k_nearest_neighbor(k: int, class_col: str, train_set: pd.DataFrame, test_set: pd.DataFrame) -> pd.DataFrame:
        LOG.info(f"Running prediction by majority label on DataFrame column {class_col}")
        if class_col not in train_set.columns:
            raise ValueError(f'Column missing from dataframe: {class_col}')
        # Run prediction
        predicted_classes = []
        for test_row_index in test_set.index:
            neighbors = []
            for train_row_index in train_set.index:
                euclidean_dist = np.linalg.norm(
                    test_set.loc[test_row_index].drop([class_col])
                    - train_set.loc[train_row_index].drop([class_col])
                )
                # Lazy method for tracking distance of K NN and associated label. Could be improved with min heap
                if len(neighbors) < k: 
                    neighbors.append([euclidean_dist, train_set.loc[train_row_index][class_col]])
                else:
                    for neighbor_index in range(len(neighbors)):
                        if neighbors[neighbor_index][0] > euclidean_dist:
                            neighbors[neighbor_index][0] = euclidean_dist
                            neighbors[neighbor_index][1] = train_set.loc[train_row_index][class_col]
                            break
            predicted_classes.append(mode([neighbor[1] for neighbor in neighbors]))
        test_set["predicted_class"] = pd.Series(predicted_classes)
        return test_set

    @staticmethod
    def edited_k_nearest_neighbor(class_col: str, train_set: pd.DataFrame, test_set: pd.DataFrame) -> pd.DataFrame:
        LOG.info(f"Running prediction by majority label on DataFrame column {class_col}")
        if class_col not in train_set.columns:
            raise ValueError(f'Column missing from dataframe: {class_col}')
        # Run prediction
        return test_set

    @staticmethod
    def condensed_k_nearest_neighbor(class_col: str, train_set: pd.DataFrame, test_set: pd.DataFrame) -> pd.DataFrame:
        LOG.info(f"Running prediction by majority label on DataFrame column {class_col}")
        if class_col not in train_set.columns:
            raise ValueError(f'Column missing from dataframe: {class_col}')
        # Run prediction
        return test_set