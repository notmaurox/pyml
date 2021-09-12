import pandas as pd
import numpy as np

from typing import Dict
# NOTE: When pulling a column out of a DataFrame, it is of type pandas.core.series.Series. This is why many of these
# methods accept that type input parameter(s).

class MetricsEvaluator(object):

    @staticmethod
    def calculate_classification_score(labels: pd.Series, predictions: pd.Series) -> float:
        if len(labels) != len(predictions):
            raise ValueError("Series are different lengths")
        labels.name = "label"
        predictions.name = "prediction"
        df = labels.to_frame().join(predictions)
        correct_predictions = len(df[df["label"] == df["prediction"]])
        return float(correct_predictions / len(df))

    @staticmethod
    def calculate_mean_squared_error(labels: pd.Series, predictions: pd.Series) -> float:
        if len(labels) != len(predictions):
            raise ValueError("Series are different lengths")
        labels.name = "label"
        predictions.name = "prediction"
        df = labels.to_frame().join(predictions)
        df["diff"] = df["label"] - df["prediction"]
        return (df["diff"].values ** 2).mean()

    @staticmethod
    def calculate_precision(labels: pd.Series, predictions: pd.Series) -> Dict:
        return_dict = {}
        if len(labels) != len(predictions):
            raise ValueError("Series are different lengths")
        labels.name = "label"
        predictions.name = "prediction"
        df = labels.to_frame().join(predictions)
        classes = pd.concat([df["label"], df["prediction"]]).unique()
        for class_name in classes:
            # Precision = TruePositives / (TruePositives + FalsePositives)
            true_positives = len(df[(df["label"] == class_name) & (df["prediction"] == class_name)])
            true_and_false_positives = (
                true_positives
                + len(df[(df["label"] != class_name) & (df["prediction"] == class_name)])
            )
            if true_and_false_positives != 0:
                return_dict[class_name] = float(true_positives / true_and_false_positives)
            else:
                return_dict[class_name] = None
        return return_dict

    @staticmethod
    def calculate_recall(labels: pd.Series, predictions: pd.Series) -> Dict:
        return_dict = {}
        if len(labels) != len(predictions):
            raise ValueError("Series are different lengths")
        labels.name = "label"
        predictions.name = "prediction"
        df = labels.to_frame().join(predictions)
        classes = pd.concat([df["label"], df["prediction"]]).unique()
        for class_name in classes:
            # Recall = TruePositives / (TruePositives + FalseNegatives)
            true_positives = len(df[(df["label"] == class_name) & (df["prediction"] == class_name)])
            true_pos_and_false_negs = (
                true_positives
                + len(df[(df["label"] == class_name) & (df["prediction"] != class_name)])
            )
            if true_pos_and_false_negs != 0:
                return_dict[class_name] = float(true_positives / true_pos_and_false_negs)
            else:
                return_dict[class_name] = None
        return return_dict

    @staticmethod
    def calculate_f1_score(labels: pd.Series, predictions: pd.Series) -> Dict:
        return_dict = {}
        class_precis_dict = MetricsEvaluator.calculate_precision(labels, predictions)
        class_recall_dict = MetricsEvaluator.calculate_recall(labels, predictions)
        for class_name in class_precis_dict.keys():
            precis = class_precis_dict[class_name]
            recall = class_recall_dict[class_name]
            if precis and recall:
                # F-Measure = (2 * Precision * Recall) / (Precision + Recall)
                return_dict[class_name] = float(2 * ((precis * recall) / (precis + recall)))
            else:
                return_dict[class_name] = None
        return return_dict

    @staticmethod
    def calculate_mean_abs_error(labels: pd.Series, predictions: pd.Series) -> float:
        if len(labels) != len(predictions):
            raise ValueError("Series are different lengths")
        labels.name = "label"
        predictions.name = "prediction"
        df = labels.to_frame().join(predictions)
        df["diff"] = df["label"] - df["prediction"]
        return (df["diff"].abs().values).mean()

    @staticmethod
    def calculate_r_sqrd_coefficient(labels: pd.Series, predictions: pd.Series) -> float:
        # TODO
        # if len(labels) != len(predictions):
        #     raise ValueError("Series are different lengths")
        # labels.name = "label"
        # predictions.name = "prediction"
        # df = labels.to_frame().join(predictions)
        # label_mean = df["label"].values.mean()
        # df["ssres"] = (df["label"] - df["prediction"])**2
        # df["sstot"] = (df["label"] - label_mean)**2
        # print(df)
        # if df["sstot"].sum() == 0:
        #     return None
        # return 1 - float(df["ssres"].sum() / df["sstot"].sum())
        return 0.0

    @staticmethod
    def calculate_pearsons_correlation(labels: pd.Series, predictions: pd.Series) -> float:
        # TODO
        # covariance(X, Y) / (stdv(X) * stdv(Y)
        if len(labels) != len(predictions):
            raise ValueError("Series are different lengths")
        return 0.0
        