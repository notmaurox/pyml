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
from learning_algorithms.CART_decision_tree_predictor import RegressionTree

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
LOG.addHandler(handler)

if __name__ == "__main__":
    data_set_name = sys.argv[1]
    if "abalone" in data_set_name:
        LOG.info("Running majority prediction on abalone data set")
        df = DataLoader.load_abalone_data()
        class_col = "rings" #attempt to predict weight of meat
    elif "forest" in data_set_name:
        LOG.info("Running majority prediction on forest fire data set")
        df = DataLoader.load_forestfires_data()
        df["area"] = df["area"] + 0.0000001 # to remove -inf from log transform
        df = DataTransformer.log_transform_column(df, "area")
        class_col = "area" #attempt to predict burned area of forest
    elif "machine" in data_set_name or "computer" in data_set_name or "hardware" in data_set_name:
        LOG.info("Running majority prediction on computer hardware data set")
        df = DataLoader.load_machine_data()
        class_col = "ERP" #attempt to predict published relative performance
    else:
        raise ValueError(f"Specified data set {data_set_name} no allowed")

    num_folds = 5
    LOG.info(f"Using {class_col} attribute as class label")
    LOG.info("Partitioning k-fold validation sets...")

    # # Discretize class into 100 buckets to help partitioning4
    df["class_discretize"] = df[class_col]
    df = DataTransformer.discretize_col(df, "class_discretize", 100)
    folds, hyperparam_set_indicies = DataTransformer.produce_k_fold_cross_validation_sets(
        df, num_folds, "class_discretize", make_hyperparam_set=True, hyperparam_set_proportion=0.2
    )
    LOG.info(f"{len(hyperparam_set_indicies)} data points set aside for hyper param tuning")
    # Commented out after performing testing of sigma's
    LOG.info("Partitioning hyperparam validation set...")
    hyper_param_fold, _ = DataTransformer.produce_k_fold_cross_validation_sets(
        df.loc[hyperparam_set_indicies], 5, "class_discretize"
    )
    df = df.drop(columns="class_discretize")
    LOG.info(f"Tuning hyper param on {len(hyper_param_fold[0][0])} training points and {len(hyper_param_fold[0][1])} test points")
    best_mse, best_allowed_mse, best_allowed_err, best_allowed_perc = float("inf"), 0, 0, 0
    for error_percent in [0.00, 0.05, 0.1, 0.15, 0.20, 0.25]:
        LOG.info(f"Using allowed error of {error_percent} of class mean for maximum allowed parition mse")
        allowed_mse = (df[class_col].mean() * error_percent)**2
        LOG.info(f"Testing allowed mse on hyperparam validation set: {allowed_mse}")
        train_df = df.loc[hyper_param_fold[0][0]].copy()
        test_df = df.loc[hyper_param_fold[0][1]].copy()
        regression_tree = RegressionTree(train_df, class_col, allowed_mse)
        regression_tree.build_tree()
        classified_tests = regression_tree.classify_examples(test_df)
        mse = MetricsEvaluator.calculate_mean_squared_error(classified_tests[class_col], classified_tests["prediction"])
        if best_mse > mse:
            best_mse = mse
            best_allowed_mse = allowed_mse
            best_allowed_err = (df[class_col].mean() * error_percent)
            best_allowed_perc = error_percent
        LOG.info(f"Parition max mse threshold | {allowed_mse} | had prediction mse on hyperparam test set | {mse}")
        LOG.info(f"Parition max mse threshold | {allowed_mse} | resulted in a tree with node count {len(regression_tree.node_store)}")
    LOG.info(f"Best mse threshold: {best_allowed_mse} had mse: {best_mse} on hyperparam set")
    prediction_scores, mse, node_count = [], [], []
    for train_indicies, test_indicies in folds:
        LOG.info(f"Learning on new fold...")
        LOG.info(f"Training on {len(train_indicies)} entitites and testing on {len(test_indicies)} entitites...")
        train_df = df.loc[train_indicies].copy()
        # LOG.info(f"Train set has class representation..\n{train_df[class_col].value_counts()}")
        test_df = df.loc[test_indicies].copy()
        # Do classification...
        regression_tree = RegressionTree(train_df, class_col, allowed_mse)
        LOG.info("Building tree...")
        regression_tree.build_tree()
        LOG.info(f"Resulting tree had {len(regression_tree.node_store)} nodes...")
        node_count.append(len(regression_tree.node_store))
        classified_tests = regression_tree.classify_examples(test_df)
        print(classified_tests)
        LOG.info(f"Calculating MSE and classification accuracy with allowed error {best_allowed_err} ({best_allowed_perc}%)")
        mse_score = MetricsEvaluator.calculate_mean_squared_error(classified_tests[class_col], classified_tests["prediction"])
        mse.append(mse_score)
        score = MetricsEvaluator.calculate_classification_score(classified_tests[class_col], classified_tests["prediction"], best_allowed_err)
        score = round(score, 4)
        prediction_scores.append(score)
        LOG.info(f"Fold had classification accuracy of {score} and MSE of {mse_score}")
    LOG.info(f"Finished 5 fold cross validation")
    LOG.info(f"Average tree node count across k-fold cross validation: {statistics.fmean(node_count)}")
    LOG.info(f"Average classification score across k-fold cross validation: {statistics.fmean(prediction_scores)}")
    LOG.info(f"Average mean squared error across k-fold cross validation: {statistics.fmean(mse)}")
