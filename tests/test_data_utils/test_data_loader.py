import unittest
import sys
import os
import pathlib

PATH_TO_SRC_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), "../", "../", "src/")
sys.path.insert(0, PATH_TO_SRC_DIR)

from data_utils.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):

    def test_load_abalone_data(self):
        df = DataLoader.load_abalone_data()
        self.assertEqual(len(df.columns), 11)

    def test_load_breast_cancer_data(self):
        df = DataLoader.load_breast_cancer_data()
        self.assertEqual(len(df.columns), 11)

    def test_load_car_data(self):
        df = DataLoader.load_car_data()
        self.assertEqual(len(df.columns), 14)

    def test_load_forestfires_data(self):
        df = DataLoader.load_forestfires_data()
        self.assertEqual(len(df.columns), 30)

    def test_load_house_votes_data(self):
        df = DataLoader.load_house_votes_data()
        self.assertEqual(len(df.columns), 50)

    def load_machine_data(self):
        df = DataLoader.load_machine_data()
        self.assertEqual(len(df.columns), 10)


        