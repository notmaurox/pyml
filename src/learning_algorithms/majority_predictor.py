import logging
import sys
import pandas as pd

# Logging stuff
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)
# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# LOG.addHandler(handler)

class MajorityPredictor(object):

    @staticmethod
    def predict_by_majority(class_col: str, train_set: pd.DataFrame, test_set: pd.DataFrame) -> pd.DataFrame:
        LOG.info(f"Running prediction by majority label on DataFrame column {class_col}")
        if class_col not in train_set.columns:
            raise ValueError(f'Column missing from dataframe: {class_col}')
        majority_label = train_set[class_col].value_counts().index[0]
        test_set["predicted_class"] = majority_label
        return test_set