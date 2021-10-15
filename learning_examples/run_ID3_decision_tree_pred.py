import sys
import os
import pathlib
import logging
import statistics
import pandas as pd

PATH_TO_SRC_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), "../", "src/")
sys.path.insert(0, PATH_TO_SRC_DIR)

from data_utils.data_loader import DataLoader
from data_utils.data_transformer import DataTransformer
from data_utils.metrics_evaluator import MetricsEvaluator
from learning_algorithms.ID3_decision_tree_predictor import ID3ClassificationTree

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
LOG.addHandler(handler)

if __name__ == "__main__":
    data_set_name = sys.argv[1]
    if "breast" in data_set_name or "cancer" in data_set_name:
        LOG.info("Running majority prediction on breast cancer data set")
        df = DataLoader.load_breast_cancer_data()
        df = df.drop(columns="sample")
        class_col = "class" #attempt to predict cancer class
    elif "car" in data_set_name:
        LOG.info("Running majority prediction on car data set")
        df = DataLoader.load_car_data()
        class_col = "acceptibility" #attempt to precit acceptibility
    elif "house" in data_set_name:
        LOG.info("Running majority prediction on house votes 84 data set")
        df = DataLoader.load_house_votes_data()
        class_col = "crime_y" #attempt to predict if vote yes on crime bill
    else:
        raise ValueError(f"Specified data set {data_set_name} no allowed")

    num_folds = 5
    LOG.info(f"Using {class_col} attribute as class label")
    LOG.info("Partitioning k-fold validation sets...")
    folds, hyperparam_set_indicies = DataTransformer.produce_k_fold_cross_validation_sets(
        df, num_folds, class_col, make_hyperparam_set=False, hyperparam_set_proportion=0.2
    )
    for train_indicies, test_indicies in folds:
        LOG.info(f"Learning on new fold...")
        LOG.info(f"Training on {len(train_indicies)} entitites and testing on {len(test_indicies)} entitites...")
        train_df = df.loc[train_indicies].copy()
        # LOG.info(f"Train set has class representation..\n{train_df[class_col].value_counts()}")
        test_df = df.loc[test_indicies].copy()
        # Do classification...