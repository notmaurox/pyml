import pathlib
import os

import pandas as pd

PATH_TO_DATA_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), "../", "../", "data/")

# 1.1
# Load data from files on disk into pandas data frames and clean them for downstream analysis
class DataLoader(object):

    @staticmethod
    def load_abalone_data() -> pd.DataFrame:
        data_file = os.path.join(PATH_TO_DATA_DIR, "abalone.data")
        header = [
            "sex", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight",
            "shell_weight", "rings"
        ]
        df = pd.read_csv(data_file, names=header)
        # Apply data transformations so that it's ready for ML application
        return df

    @staticmethod
    def load_breast_cancer_data() -> pd.DataFrame:
        data_file = os.path.join(PATH_TO_DATA_DIR, "breast-cancer-wisconsin.data")
        header = [
            "sample", "clump_thickness", "uniformity_of_cell_size", "uniformity_of_cell_shape",
            "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei",
            "bland_chromatin", "normal_nucleoli", "mitoses", "class"
        ]
        df = pd.read_csv(data_file, names=header)
        # Apply data transformations so that it's ready for ML application
        return df

    @staticmethod
    def load_car_data() -> pd.DataFrame:
        data_file = os.path.join(PATH_TO_DATA_DIR, "car.data")
        header = [
            "buying", "maint", "doors", "persons", "lug_boot", "safety"
        ]
        df = pd.read_csv(data_file, names=header)
        # Apply data transformations so that it's ready for ML application
        return df

    @staticmethod
    def load_forestfires_data() -> pd.DataFrame:
        data_file = os.path.join(PATH_TO_DATA_DIR, "forestfires.data")
        header = [
            "x", "y", "month", "day", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain", "area"
        ]
        df = pd.read_csv(data_file, names=header)
        # Apply data transformations so that it's ready for ML application
        return df


    @staticmethod
    def load_house_votes_data() -> pd.DataFrame:
        data_file = os.path.join(PATH_TO_DATA_DIR, "house-votes-84.data")
        header = [
            "class_name", "handicapped_infants", "water_project_cost_sharing", "adoption_of_budget_resoluiton",
            "physician_fee_freeze", "el_salvador_aid", "religious_groups_in_schools", "anti_satellite_test_ban",
            "aid_to_nicaraguan_contras", "mx_missle", "immigration", "synthfuels_corportation_cutback",
            "education_spending", "superfund_right_to_sue", "crime", "duty_free_exports",
            "export_admin_act_south_africa"
        ]
        df = pd.read_csv(data_file, names=header)
        # Apply data transformations so that it's ready for ML application
        return df

    @staticmethod
    def load_machine_data() -> pd.DataFrame:
        data_file = os.path.join(PATH_TO_DATA_DIR, "house-votes-84.data")
        header = [
            "vendor", "model_name", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"
        ]
        df = pd.read_csv(data_file, names=header)
        # Apply data transformations so that it's ready for ML application
        return df

if __name__ == "__main__":
    df = DataLoader.load_forestfires_data()
    print(df.columns)
