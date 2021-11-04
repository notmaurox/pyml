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
from learning_algorithms.logistic_classifier import LogisticRegressionClassifier

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
LOG.addHandler(handler)

DO_PRUNING = False

# Takes two runtime args...
# 1 - Name of data set to use..
# 2 - true if to do pruning, false if not.

if __name__ == "__main__":
    data_set_name = sys.argv[1]
    if "breast" in data_set_name or "cancer" in data_set_name:
        LOG.info("Running majority prediction on breast cancer data set")
        df = DataLoader.load_breast_cancer_data()
        df = df.drop(columns="sample")
        class_col = "class" #attempt to predict cancer class
        for column_name in df.columns:
            if column_name == class_col:
                continue
            df = DataTransformer.handle_nomal_col(df, column_name)
    elif "car" in data_set_name:
        LOG.info("Running majority prediction on car data set")
        df = DataLoader.load_car_data()
        class_col = "acceptibility" #attempt to precit acceptibility
        for column_name in df.columns:
            if column_name == class_col:
                continue
            df = DataTransformer.handle_nomal_col(df, column_name)
    elif "house" in data_set_name:
        LOG.info("Running majority prediction on house votes 84 data set")
        df = DataLoader.load_house_votes_data()
        class_col = "crime_y" #attempt to predict if vote yes on crime bill
    else:
        raise ValueError(f"Specified data set {data_set_name} no allowed")
    LOG.info(f"Using {class_col} attribute as class label")
    num_folds = 5
    print(df)
    LOG.info("Partitioning k-fold validation sets...")
    folds, hyperparam_set_indicies = DataTransformer.produce_k_fold_cross_validation_sets(
        df, num_folds, class_col, make_hyperparam_set=False, hyperparam_set_proportion=0.2
    )
    prediction_scores, pruned_prediction_scores, data_iterations = [], [], []
    for train_indicies, test_indicies in folds:
        LOG.info(f"Learning on new fold with {len(train_indicies)} entitites and testing on {len(test_indicies)} entitites...")
        train_df = df.loc[train_indicies].copy()
        # LOG.info(f"Train set has class representation..\n{train_df[class_col].value_counts()}")
        test_df = df.loc[test_indicies].copy()
        # Build tree w/o pruning...
        logistic_regression_classifier = LogisticRegressionClassifier(train_df, class_col)
        iterations = logistic_regression_classifier.learn(allowed_bad_iterations=1)
        classified_examples = logistic_regression_classifier.classify_examples(test_df)
        score = MetricsEvaluator.calculate_classification_score(classified_examples[class_col], classified_examples["prediction"])
        score = round(score, 4)
        prediction_scores.append(score)
        data_iterations.append(iterations)
        LOG.info(f"Fold had classification accuracy of {score} on test set")
    LOG.info(f"Finished 5 fold cross validation")
    LOG.info(f"Average classification score across k-fold cross validation on test sets: {statistics.fmean(prediction_scores)}")
    LOG.info(f"Average number of iterations through training set: {statistics.fmean(data_iterations)}")