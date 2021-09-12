import pandas as pd

class MajorityPredictor(object):

    @staticmethod
    def predict_by_majority(class_col: str, train_set: pd.DataFrame, test_set: pd.DataFrame) -> pd.DataFrame:
        if class_col not in train_set.columns:
            raise ValueError(f'Column missing from dataframe: {class_col}')
        majority_label = train_set[class_col].value_counts().index[0]
        test_set["predicted_class"] = majority_label
        return test_set