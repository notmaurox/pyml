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
from learning_algorithms.k_nn_predictor import KNearestNeighborPredictor

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
LOG.addHandler(handler)

if __name__ == "__main__":
    data_set_name = sys.argv[1]
    k = int(sys.argv[2])
    nn_type = sys.argv[3]
    if "abalone" in data_set_name:
        LOG.info("Running majority prediction on abalone data set")
        df = DataLoader.load_abalone_data()
        class_col = "rings" #attempt to predict weight of meat
        do_regression = True
        sigma = 0.5
    elif "breast" in data_set_name or "cancer" in data_set_name:
        LOG.info("Running majority prediction on breast cancer data set")
        df = DataLoader.load_breast_cancer_data()
        df = df.drop(columns="sample")
        class_col = "class" #attempt to predict cancer class
        do_regression = False
    elif "car" in data_set_name:
        LOG.info("Running majority prediction on car data set")
        df = DataLoader.load_car_data()
        class_col = "acceptibility" #attempt to precit acceptibility
        do_regression = False
    elif "forest" in data_set_name:
        LOG.info("Running majority prediction on forest fire data set")
        df = DataLoader.load_forestfires_data()
        df["area"] = df["area"] + 0.0000001 # to remove -inf from log transform
        df = DataTransformer.log_transform_column(df, "area")
        class_col = "area" #attempt to predict burned area of forest
        do_regression = True
        sigma = 10000
    elif "house" in data_set_name:
        LOG.info("Running majority prediction on house votes 84 data set")
        df = DataLoader.load_house_votes_data()
        class_col = "crime_y" #attempt to predict if vote yes on crime bill
        do_regression = False
    elif "machine" in data_set_name or "computer" in data_set_name or "hardware" in data_set_name:
        LOG.info("Running majority prediction on computer hardware data set")
        df = DataLoader.load_machine_data()
        class_col = "ERP" #attempt to predict published relative performance
        do_regression = True
        sigma = 100
    else:
        raise ValueError(f"Specified data set {data_set_name} no allowed")
    print(df)
    num_folds = 5
    LOG.info(f"Using {class_col} attribute as class label")
    allowed_error = None
    if do_regression:
        allowed_error = df[class_col].mean() * 0.15
        LOG.info(f"Allowing 10% error in classification means allowance of {str(allowed_error)}")
    knnp = KNearestNeighborPredictor(
        k, do_regression, sigma=2.5, allowed_error=allowed_error
    )
    if knnp.do_regression:
        LOG.info("Partitioning k-fold validation sets...")
        df["class_discretize"] = df[class_col]
        # # Discretize class into 100 buckets to help partitioning
        df = DataTransformer.discretize_col(df, "class_discretize", 100)
        folds, hyperparam_set_indicies = DataTransformer.produce_k_fold_cross_validation_sets(
            df, num_folds, "class_discretize", make_hyperparam_set=True, hyperparam_set_proportion=0.2
        )
        LOG.info(f"{len(hyperparam_set_indicies)} data points set aside for hyper param tuning")
        sigmas_to_test = [10000, 100000, 1000000, 10000000, 100000000]
        best_mse = float('inf')
        best_sigma = sigma
        knnp.sigma = best_sigma
        # Commented out after performing testing of sigma's
        LOG.info("Partitioning hyperparam validation set...")
        hyper_param_fold, _ = DataTransformer.produce_k_fold_cross_validation_sets(
            df.loc[hyperparam_set_indicies], 5, "class_discretize"
        )
        df = df.drop(columns="class_discretize")
        # LOG.info(f"Tuning hyper param on {len(hyper_param_fold[0][0])} training points and {len(hyper_param_fold[0][1])} test points")
        for sigma in sigmas_to_test:
            LOG.info(f"Testing sigma param: {sigma}")
            knnp.sigma = sigma
            train_df = df.loc[hyper_param_fold[0][0]].copy()
            test_df = df.loc[hyper_param_fold[0][1]].copy()
            predictions = knnp.k_nearest_neighbor(class_col, train_df, test_df)
            mse = MetricsEvaluator.calculate_mean_squared_error(predictions[class_col], predictions["prediction"])
            if best_mse > mse:
                best_mse = mse
                best_sigma = sigma
            LOG.info(f"sigma param: {sigma} had mse: {mse} on hyperparam set")
        LOG.info(f"Best sigma param: {best_sigma} had mse: {best_mse} on hyperparam set")
    else:
        LOG.info("Partitioning k-fold validation sets...")
        folds, hyperparam_set_indicies = DataTransformer.produce_k_fold_cross_validation_sets(
            df, num_folds, class_col
        )
    prediction_scores = []
    mse = []
    for train_indicies, test_indicies in folds:
        # LOG.info(f"Training on {len(train_indicies)} entitites and testing on {len(test_indicies)} entitites")
        train_df = df.loc[train_indicies].copy()
        # LOG.info(f"Train set has class representation..\n{train_df[class_col].value_counts()}")
        test_df = df.loc[test_indicies].copy()
        # Do classification...
        if "edited" in nn_type:
            LOG.info("Performing classification on fold with edited k nn...")
            pred_df = knnp.edited_k_nearest_neighbor(class_col, train_df, test_df)
        elif "condensed" in nn_type:
            LOG.info("Performing classification on fold with condensed k nn...")
            pred_df = knnp.condensed_k_nearest_neighbor(class_col, train_df, test_df)
        else:
            LOG.info("Performing classification on fold with k nn...")
            pred_df = knnp.k_nearest_neighbor(class_col, train_df, test_df)
        print(pred_df)
        # Check how well we did....
        if do_regression:
            LOG.info(f"Calculating MSE and classification accuracy with allowed error {allowed_error}")
            mse_score = MetricsEvaluator.calculate_mean_squared_error(pred_df[class_col], pred_df["prediction"])

            mse.append(mse_score)
            score = MetricsEvaluator.calculate_classification_score(pred_df[class_col], pred_df["prediction"], allowed_error)
            score = round(score, 4)
            prediction_scores.append(score)
            LOG.info(f"Fold had classification accuracy of {score} and MSE of {mse_score}")
        else:
            LOG.info("Calculating classification accuracy")
            score = MetricsEvaluator.calculate_classification_score(pred_df[class_col], pred_df["prediction"])
            prediction_scores.append(score)
            LOG.info(f"Fold had score of {score}")
        

    LOG.info(f"Average classification score across k-fold cross validation: {statistics.fmean(prediction_scores)}")
    if do_regression:
        LOG.info(f"Average mean squared error across k-fold cross validation: {statistics.fmean(mse)}")
