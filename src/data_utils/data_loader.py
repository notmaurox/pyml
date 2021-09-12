import pathlib
import os
import sys
import logging

import pandas as pd

PATH_TO_SRC_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), "../", "src/")
sys.path.insert(0, PATH_TO_SRC_DIR)

from .data_transformer import DataTransformer

PATH_TO_DATA_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), "../", "../", "data/")

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

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
        df = DataTransformer.handle_nomal_col(df, "sex")
        for column_name in DataTransformer.identify_cols_with_missing_data(df):
            LOG.info(f"filling in missing vals in {column_name} with avrg")
            df = DataTransformer.impute_missing_vales_with_mean(df, column_name)
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
        for column_name in DataTransformer.identify_cols_with_missing_data(df):
            LOG.info(f"filling in missing vals in {column_name} with avrg")
            df = DataTransformer.impute_missing_vales_with_mean(df, column_name)
        # Apply data transformations so that it's ready for ML application
        return df

    @staticmethod
    def load_car_data() -> pd.DataFrame:
        data_file = os.path.join(PATH_TO_DATA_DIR, "car.data")
        header = [
            "buying", "maint", "doors", "persons", "lug_boot", "safety", "acceptibility",
        ]
        # this data doesnt come with row idenfiers so index_col set to false
        df = pd.read_csv(data_file, names=header, index_col=False)
        # Normalize string fields with relatinships such that lower numbers are better
        df = DataTransformer.handle_ordinal_col_with_map(
            df, "buying",
            {
                "low": 0,
                "med": 1,
                "high": 2,
                "vhigh": 3
            }
        )
        df = DataTransformer.handle_ordinal_col_with_map(
            df, "maint",
            {
                "low": 0,
                "med": 1,
                "high": 2,
                "vhigh": 3
            }
        )
        df = DataTransformer.handle_ordinal_col_with_map(
            df, "safety",
            {
                "high": 0,
                "med": 1,
                "low": 2
            }
        )
        df = DataTransformer.handle_ordinal_col_with_map(
            df, "acceptibility",
            {
                "v-good": 0,
                "good": 1,
                "acc": 2,
                "unacc": 3
            }
        )
        # For these fields, it's less clear the way in which fields are ordered
        df = DataTransformer.handle_nomal_col(df, "doors")
        df = DataTransformer.handle_nomal_col(df, "persons")
        df = DataTransformer.handle_nomal_col(df, "lug_boot")
        # fill in missing data
        for column_name in DataTransformer.identify_cols_with_missing_data(df):
            LOG.info(f"filling in missing vals in {column_name} with avrg")
            df = DataTransformer.impute_missing_vales_with_mean(df, column_name)
        return df

    @staticmethod
    def load_forestfires_data() -> pd.DataFrame:
        data_file = os.path.join(PATH_TO_DATA_DIR, "forestfires.data")
        header = [
            "x", "y", "month", "day", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain", "area"
        ]
        df = pd.read_csv(data_file, names=header)
        # break month and day fields into discrente dimensions
        df = DataTransformer.handle_nomal_col(df, "month")
        df = DataTransformer.handle_nomal_col(df, "day")
        # fill in missing data
        for column_name in DataTransformer.identify_cols_with_missing_data(df):
            LOG.info(f"filling in missing vals in {column_name} with avrg")
            df = DataTransformer.impute_missing_vales_with_mean(df, column_name)
        # df = DataTransformer.log_transform_column(df, "area")
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
        # Here pretty much every column should be broken up as nominal features
        for column_name in header:
            df = DataTransformer.handle_nomal_col(df, column_name)
        return df

    @staticmethod
    def load_machine_data() -> pd.DataFrame:
        data_file = os.path.join(PATH_TO_DATA_DIR, "machine.data")
        header = [
            "vendor", "model_name", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"
        ]
        df = pd.read_csv(data_file, names=header)
        # Apply data transformations so that it's ready for ML application
        df = DataTransformer.handle_nomal_col(df, "vendor")
        return df
