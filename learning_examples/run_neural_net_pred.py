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
from learning_algorithms.neural_net_prediction import NeuralNetwork, Autoencoder

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
LOG.addHandler(handler)

if __name__ == "__main__":
    data_set_name = sys.argv[1]
    autoencode = sys.argv[2]
    # REGRESSION data sets
    if "abalone" in data_set_name:
        LOG.info("Running majority prediction on abalone data set")
        df = DataLoader.load_abalone_data()
        class_col = "rings" #attempt to predict weight of meat
        for column_name in ["length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight"]:
            if column_name == class_col:
                continue
            df = DataTransformer.min_max_normalize_column(df, column_name)
        do_regression = True
        learning_rate = 0.2
        max_iterations = 100
    elif "forest" in data_set_name:
        # "x", "y", "month", "day", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain", "area"
        LOG.info("Running majority prediction on forest fire data set")
        df = DataLoader.load_forestfires_data()
        df["area"] = df["area"] + 0.01 # to remove -inf from log transform
        df = DataTransformer.log_transform_column(df, "area")
        df = df.drop(columns="x").drop(columns="y")
        class_col = "area" #attempt to predict burned area of forest
        for column_name in ["FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]:
            df = DataTransformer.min_max_normalize_column(df, column_name)
        do_regression = True
        learning_rate = 0.00001
        max_iterations = 200
    elif "machine" in data_set_name or "computer" in data_set_name or "hardware" in data_set_name:
        LOG.info("Running majority prediction on computer hardware data set")
        df = DataLoader.load_machine_data()
        class_col = "ERP" #attempt to predict published relative performance
        for column_name in df.columns:
            if column_name == class_col:
                continue
            df = DataTransformer.min_max_normalize_column(df, column_name)
        do_regression = True
        learning_rate = 0.0001
        max_iterations = 1200
    # CLASSIFICATION DATASETS
    elif "breast" in data_set_name or "cancer" in data_set_name:
        LOG.info("Running majority prediction on breast cancer data set")
        df = DataLoader.load_breast_cancer_data()
        df = df.drop(columns="sample")
        class_col = "class" #attempt to predict cancer class
        for column_name in df.columns:
            if column_name == class_col:
                continue
            df = DataTransformer.min_max_normalize_column(df, column_name)
        do_regression = False
        learning_rate = 0.8
        max_iterations = 100
    elif "car" in data_set_name:
        LOG.info("Running majority prediction on car data set")
        df = DataLoader.load_car_data()
        class_col = "acceptibility" #attempt to predict acceptibility
        # Doors and persons should be nominal
        # safety, lug_boot, maint, buying should be min max normalized...
        for column_name in ["buying", "maint", "safety"]:
            df = DataTransformer.min_max_normalize_column(df, column_name)
        do_regression = False
        learning_rate = 0.80
        max_iterations = 150
    elif "house" in data_set_name:
        LOG.info("Running majority prediction on house votes 84 data set")
        df = DataLoader.load_house_votes_data(adjust=False)
        class_col = "class_name" #attempt to predict class affiliation
        for col in df.columns:
            if col == class_col:
                continue
            df = DataTransformer.handle_nomal_col(df, col)
        do_regression = False
        learning_rate = 0.25
        max_iterations = 800
    else:
        raise ValueError(f"Specified data set {data_set_name} no allowed")
    num_folds = 5
    LOG.info(f"Using {class_col} attribute as class label")
    LOG.info("Partitioning k-fold validation sets...")
    if do_regression: # Discretize class into 100 buckets to help partitioning
        df["class_discretize"] = df[class_col]
        df = DataTransformer.discretize_col(df, "class_discretize", 100)
        folds, hyperparam_set_indicies = DataTransformer.produce_k_fold_cross_validation_sets(
            df, num_folds, "class_discretize", make_hyperparam_set=False, hyperparam_set_proportion=0.2
        )
        df = df.drop(columns="class_discretize")
        allowed_err = (df[class_col].mean() * 0.15)
    else:
        folds, hyperparam_set_indicies = DataTransformer.produce_k_fold_cross_validation_sets(
            df, num_folds, class_col, make_hyperparam_set=False, hyperparam_set_proportion=0.2
        )
    print(df)
    prediction_scores, mse, iterations = [], [], []
    fold_count = 1
    for train_indicies, test_indicies in folds:
        LOG.info(f"Learning on new fold...")
        LOG.info(f"Training on {len(train_indicies)} entitites and testing on {len(test_indicies)} entitites...")
        train_df = df.loc[train_indicies].copy()
        test_df = df.loc[test_indicies].copy()
        nn = NeuralNetwork(train_df, class_col, 2, 7, do_regression, learning_rate)
        if autoencode != "false":
            LOG.info(f"Training Autoencoder...")
            ac = Autoencoder(train_df, class_col, 5, learning_rate)
            ac.train_network(25)
            LOG.info(f"Applying autoencoder layer to NN...")
            nn.apply_autoencoder_layer(ac.layers[0])
        training_iterations = nn.train_network(max_iterations)
        iterations.append(training_iterations + 1) # Index to count conversion...
        LOG.info(f"Training stopped after {training_iterations} iterations")
        classified_tests = nn.classify_examples(test_df)
        print(classified_tests[[class_col, "prediction"]])
        if do_regression:
            mse_score = MetricsEvaluator.calculate_mean_squared_error(classified_tests[class_col], classified_tests["prediction"])
            mse.append(mse_score)
            plt.scatter(classified_tests[class_col], classified_tests["prediction"], label=f"Fold {fold_count}", alpha=0.25)
            LOG.info(f"Fold had MSE of {mse_score}")
        else:
            score = MetricsEvaluator.calculate_classification_score(classified_tests[class_col], classified_tests["prediction"])
            score = round(score, 4)
            prediction_scores.append(score)
            print(classified_tests[class_col].unique(), classified_tests["prediction"].unique())
            LOG.info(f"Fold had classification accuracy of {score}")
        fold_count += 1
    LOG.info(f"Finished 5 fold cross validation")
    LOG.info(f"Learning rate: {learning_rate}")
    LOG.info(f"Max allowed iterations: {max_iterations}")
    LOG.info(f"Average iterations before MSE increased: {statistics.fmean(iterations)}")
    if do_regression:
        LOG.info(f"Average mean squared error across k-fold cross validation: {statistics.fmean(mse)}")
        LOG.info(f'Fold MSEs: {", ".join([str(round(item, 3)) for item in mse])}')
        plt.xlabel(f"Actual (MSE:{round(statistics.fmean(mse), 3)})")
        plt.ylabel("Predicted")
        xpoints = ypoints = plt.xlim()
        plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)
        plt.show()
    else:
        LOG.info(f"Average classification score across k-fold cross validation: {statistics.fmean(prediction_scores)}")
        LOG.info(f'Fold accuracies: {", ".join([str(round(item, 3)) for item in prediction_scores])}')

    # LOG.info(f"Average iterations before MSE increased: {statistics.fmean(iterations)}")
    # plt.xlabel(f"Actual (MSE:{round(statistics.fmean(mse), 3)})")
    # plt.ylabel("Predicted")
    # xpoints = ypoints = plt.xlim()
    # plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)
    # plt.show()
