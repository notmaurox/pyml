import sys
import os
import pathlib
import logging
import statistics
import pandas as pd
import matplotlib.pyplot as plt

PATH_TO_SRC_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), "../", "src/")
sys.path.insert(0, PATH_TO_SRC_DIR)

from data_utils.data_loader import DataLoader
from data_utils.data_transformer import DataTransformer
from data_utils.metrics_evaluator import MetricsEvaluator
from learning_algorithms.linear_regressor import LinearRegressor

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
LOG.addHandler(handler)

if __name__ == "__main__":
    data_set_name = sys.argv[1]
    min_iterations = int(sys.argv[2])
    if "abalone" in data_set_name:
        LOG.info("Running majority prediction on abalone data set")
        df = DataLoader.load_abalone_data()
        class_col = "rings" #attempt to predict weight of meat
        for column_name in df.columns:
            if column_name == class_col:
                continue
            df = DataTransformer.handle_nomal_col(df, column_name)
        print(df)
    elif "forest" in data_set_name:
        LOG.info("Running majority prediction on forest fire data set")
        df = DataLoader.load_forestfires_data()
        df["area"] = df["area"] + 0.01 # to remove -inf from log transform
        df = DataTransformer.log_transform_column(df, "area")
        class_col = "area" #attempt to predict burned area of forest
        for column_name in df.columns:
            if column_name == class_col:
                continue
            df = DataTransformer.handle_nomal_col(df, column_name)
    elif "machine" in data_set_name or "computer" in data_set_name or "hardware" in data_set_name:
        LOG.info("Running majority prediction on computer hardware data set")
        df = DataLoader.load_machine_data()
        class_col = "ERP" #attempt to predict published relative performance
        for column_name in df.columns:
            if column_name == class_col:
                continue
            df = DataTransformer.handle_nomal_col(df, column_name)
    else:
        raise ValueError(f"Specified data set {data_set_name} no allowed")
    num_folds = 5

    LOG.info(f"Using {class_col} attribute as class label")
    LOG.info("Partitioning k-fold validation sets...")
    # # Discretize class into 100 buckets to help partitioning4
    df["class_discretize"] = df[class_col]
    df = DataTransformer.discretize_col(df, "class_discretize", 100)
    folds, hyperparam_set_indicies = DataTransformer.produce_k_fold_cross_validation_sets(
        df, num_folds, "class_discretize", make_hyperparam_set=False, hyperparam_set_proportion=0.2
    )
    df = df.drop(columns="class_discretize")
    allowed_err = (df[class_col].mean() * 0.15)
    prediction_scores, mse, iterations = [], [], []
    fold_count = 1
    for train_indicies, test_indicies in folds:
        LOG.info(f"Learning on new fold...")
        LOG.info(f"Training on {len(train_indicies)} entitites and testing on {len(test_indicies)} entitites...")
        train_df = df.loc[train_indicies].copy()
        test_df = df.loc[test_indicies].copy()
        # Do classification...
        lr = LinearRegressor(train_df, class_col)
        curr_iteration, mses = lr.learn(min_iterations)
        iterations.append(curr_iteration)
        # plt.scatter([i+1 for i in range(len(mses))], mses, label=f"Fold {fold_count}")
        classified_tests = lr.classify_examples(test_df)
        plt.scatter(classified_tests[class_col], classified_tests["prediction"], label=f"Fold {fold_count}", alpha=0.25)
        print(classified_tests[[class_col, "prediction"]])
        LOG.info(f"Calculating MSE and classification accuracy with allowed error {allowed_err} (15%)")
        mse_score = MetricsEvaluator.calculate_mean_squared_error(classified_tests[class_col], classified_tests["prediction"])
        mse.append(mse_score)
        score = MetricsEvaluator.calculate_classification_score(classified_tests[class_col], classified_tests["prediction"], allowed_err)
        score = round(score, 4)
        prediction_scores.append(score)
        LOG.info(f"Fold had classification accuracy of {score} and MSE of {mse_score}")
        fold_count += 1
    LOG.info(f"Finished 5 fold cross validation")
    LOG.info(f"Average classification score across k-fold cross validation: {statistics.fmean(prediction_scores)}")
    LOG.info(f"Average mean squared error across k-fold cross validation: {statistics.fmean(mse)}")
    LOG.info(f"Average iterations before MSE increased: {statistics.fmean(iterations)}")
    plt.xlabel(f"Actual (MSE:{round(statistics.fmean(mse), 3)})")
    plt.ylabel("Predicted")
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)
    plt.show()
