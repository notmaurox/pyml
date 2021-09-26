import sys
import os
import pathlib
import logging
import statistics

PATH_TO_SRC_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), "../", "src/")
sys.path.insert(0, PATH_TO_SRC_DIR)

from data_utils.data_loader import DataLoader
from data_utils.data_transformer import DataTransformer
from data_utils.metrics_evaluator import MetricsEvaluator
from learning_algorithms.k_nn_predictor import KNearestNeighborsPredictor

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# LOG.addHandler(handler)

if __name__ == "__main__":
    data_set_name = sys.argv[1]
    if "abalone" in data_set_name:
        LOG.info("Running majority prediction on abalone data set")
        df = DataLoader.load_abalone_data()
        class_col = "shucked_weight" #attempt to predict weight of meat
        calculate_mse = True
    elif "breast" in data_set_name or "cancer" in data_set_name:
        LOG.info("Running majority prediction on breast cancer data set")
        df = DataLoader.load_breast_cancer_data()
        class_col = "class" #attempt to predict cancer class
        calculate_mse = True
    elif "car" in data_set_name:
        LOG.info("Running majority prediction on car data set")
        df = DataLoader.load_car_data()
        class_col = "acceptibility" #attempt to precit acceptibility
        calculate_mse = False
    elif "forest" in data_set_name:
        LOG.info("Running majority prediction on forest fire data set")
        df = DataLoader.load_forestfires_data()
        class_col = "area" #attempt to predict burned area of forest
        # since we are training on area, it makes sense to bin them
        df = DataTransformer.discretize_col(df, "area", 5, equal_width=False, equal_freq=True)
        calculate_mse = False
    elif "house" in data_set_name:
        LOG.info("Running majority prediction on house votes 84 data set")
        df = DataLoader.load_house_votes_data()
        class_col = "crime_y" #attempt to predict if vote yes on crime bill
        calculate_mse = False
    elif "machine" in data_set_name:
        LOG.info("Running majority prediction on machine data set")
        df = DataLoader.load_machine_data()
        class_col = "PRP" #attempt to predict published relative performance
        calculate_mse = True
    else:
        raise ValueError(f"Specified data set {data_set_name} no allowed")
    print(df)
    num_folds = 5
    folds, _ = DataTransformer.produce_k_fold_cross_validation_sets(df, num_folds, class_col)
    prediction_scores = []
    mses = []
    for train_indicies, test_indicies in folds:
        LOG.info(f"Training on {len(train_indicies)} entitites and testing on {len(test_indicies)} entitites")
        train_df = df.loc[train_indicies].copy()
        LOG.info(f"Train set has class representation..\n{train_df[class_col].value_counts()}")
        test_df = df.loc[test_indicies].copy()
        LOG.info(f"Test set has class representation..\n{test_df[class_col].value_counts()}")

        
        MajorityPredictor.predict_by_majority(class_col, train_df, test_df)
        c_score = MetricsEvaluator.calculate_classification_score(test_df[class_col], test_df["predicted_class"])
        prediction_scores.append(c_score)
        LOG.info(f"Fold had classification score: {c_score}")
        # f1_score = MetricsEvaluator.calculate_f1_score(test_df[class_col], test_df["predicted_class"])
        # LOG.info(f"Fold had class F1 scores: {f1_score}")
        if calculate_mse:
            mse = MetricsEvaluator.calculate_mean_squared_error(test_df[class_col], test_df["predicted_class"])
            LOG.info(f"Fold had mean squared error: {mse}")
            mses.append(mse)
    LOG.info(f"Average prediction score across k-fold cross validation: {statistics.fmean(prediction_scores)}")
    if calculate_mse:
        LOG.info(f"Average mean squared error across k-fold cross validation: {statistics.fmean(mses)}")
