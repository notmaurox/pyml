import unittest
import sys
import os
import pathlib

import pandas as pd

PATH_TO_SRC_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), "../", "../", "src/")
sys.path.insert(0, PATH_TO_SRC_DIR)

from learning_algorithms.ID3_decision_tree_predictor import ID3ClassificationTree

class TestID3ClassificationTree(unittest.TestCase):

    def test_handle_numeric_attributes(self):
        cols = ["example", "numeric_col", "class"]
        data = [
            [1, 1,  "Y"], #0 
            [2, 2,  "Y"], #1 
            [3, 3,  "Y"], #2 
            [4, 4,  "Y"], #0 
            [5, 5,  "Y"], #1 
            [6, 10,  "N"], #2 
            [7, 11,  "N"], # 
            [8, 12,  "N"], #0 
            [9, 13,  "N"], #1 
            [10, 20, "P"], #2 
            [11, 22, "P"], #0 
            [12, 23, "P"], #1 
            [13, 24, "P"], #2 
            [14, 25, "P"], #0 
        ]
        df = pd.DataFrame(data, columns=cols)
        df = ID3ClassificationTree.handle_numeric_attributes(df, "class")
        # After transformation, dataframe looks like...
        #     example            numeric_col       class
        # 0   Y_feat_val_bucket  Y_feat_val_bucket     Y
        # 1   Y_feat_val_bucket  Y_feat_val_bucket     Y
        # 2   Y_feat_val_bucket  Y_feat_val_bucket     Y
        # 3   Y_feat_val_bucket  Y_feat_val_bucket     Y
        # 4   Y_feat_val_bucket  Y_feat_val_bucket     Y
        # 5   N_feat_val_bucket  N_feat_val_bucket     N
        # 6   N_feat_val_bucket  N_feat_val_bucket     N
        # 7   N_feat_val_bucket  N_feat_val_bucket     N
        # 8   N_feat_val_bucket  N_feat_val_bucket     N
        # 9   P_feat_val_bucket  P_feat_val_bucket     P
        # 10  P_feat_val_bucket  P_feat_val_bucket     P
        # 11  P_feat_val_bucket  P_feat_val_bucket     P
        # 12  P_feat_val_bucket  P_feat_val_bucket     P
        # 13  P_feat_val_bucket  P_feat_val_bucket     P
        self.assertEqual(["Y_feat_val_bucket", "N_feat_val_bucket", "P_feat_val_bucket"], df["example"].unique().tolist())
        self.assertEqual(["Y_feat_val_bucket", "N_feat_val_bucket", "P_feat_val_bucket"], df["numeric_col"].unique().tolist())

    def test_calc_entropy(self):
        class_composition = {
            "pos": 9,
            "neg": 5,
        }
        total_examples = 14
        entropy = ID3ClassificationTree.calc_entropy(class_composition, total_examples)
        self.assertEqual(round(entropy, 2), 0.94)

    def test_calculate_feature_and_info_val(self):
        cols = ["example", "outlook", "class"]
        data = [
            [1, "S",  "N"], #0
            [2, "S",  "N"], #1
            [3, "O",  "P"], #2
            [4, "R",  "P"], #0
            [5, "R",  "N"], #1
            [6, "R",  "P"], #2
            [7, "O",  "P"], #
            [8, "S",  "N"], #0
            [9, "S",  "P"], #1
            [10, "R", "P"], #2
            [11, "S", "P"], #0
            [12, "O", "P"], #1
            [13, "O", "P"], #2
            [14, "R", "N"], #0
        ]
        df = pd.DataFrame(data, columns=cols)
        feat_e, gain, info_val = ID3ClassificationTree.calculate_feature_gain_and_info_val(df["outlook"], df["class"])
        self.assertEqual(round(feat_e, 3), 0.694)
        # Due to accuracy of code calculation, gain = 0.247 not 0.246 like in lecture
        self.assertEqual(round(gain, 2), 0.25)
        # Information value of Outlook = 
        # 5/14*log(5/14) - 4/14*log(0/14) - 5/14*log(5/14)
        # -(-0.531 + -0.516 + -0.531) = 1.578. Due accuracy of code calculation, info_val comes out to 1.577
        self.assertEqual(round(info_val, 3), 1.577)
        gain_ratio = gain / info_val # 0.25 / 1.577
        self.assertEqual(round(gain_ratio, 2), 0.16)

    def test_id_best_feature(self):
        cols = ["example", "outlook", "temp", "humidity", "wind", "class"]
        data = [
            [1,  "S", "H", "H", "F", "N"], #0
            [2,  "S", "H", "H", "T", "N"], #1
            [3,  "O", "H", "H", "F", "P"], #2
            [4,  "R", "M", "H", "F", "P"], #0
            [5,  "R", "C", "N", "F", "N"], #1
            [6,  "R", "C", "N", "T", "P"], #2
            [7,  "O", "C", "N", "T", "P"], #
            [8,  "S", "M", "H", "F", "N"], #0
            [9,  "S", "C", "N", "F", "P"], #1
            [10, "R", "M", "N", "F", "P"], #2
            [11, "S", "M", "N", "T", "P"], #0
            [12, "O", "M", "H", "T", "P"], #1
            [13, "O", "H", "N", "F", "P"], #2
            [14, "R", "M", "H", "T", "N"], #0
        ]
        df = pd.DataFrame(data, columns=cols)
        best_feature = ID3ClassificationTree.pick_best_feature_to_split(df.drop("example", axis=1), "class")
        self.assertEqual(best_feature, "outlook")

    def test_build_tree_and_classify(self):
        cols = ["example", "outlook", "temp", "humidity", "wind", "class"]
        data = [
            [1,  "S", "H", "H", "F", "N"], #0
            [2,  "S", "H", "H", "T", "N"], #1
            [3,  "O", "H", "H", "F", "P"], #2
            [4,  "R", "M", "H", "F", "P"], #0
            [5,  "R", "C", "N", "F", "N"], #1
            [6,  "R", "C", "N", "T", "P"], #2
            [7,  "O", "C", "N", "T", "P"], #
            [8,  "S", "M", "H", "F", "N"], #0
            [9,  "S", "C", "N", "F", "P"], #1
            [10, "R", "M", "N", "F", "P"], #2
            [11, "S", "M", "N", "T", "P"], #0
            [12, "O", "M", "H", "T", "P"], #1
            [13, "O", "H", "N", "F", "P"], #2
            [14, "R", "M", "H", "T", "N"], #0
        ]
        df = pd.DataFrame(data, columns=cols)
        # Build tree
        id3_tree = ID3ClassificationTree(df.drop("example", axis=1), "class")
        id3_tree.build_tree(id3_tree.root)
        self.assertEqual(id3_tree.root.feature, "outlook")
        self.assertEqual(id3_tree.root.children["S"].feature, "humidity")
        self.assertTrue(id3_tree.root.children["O"].is_pure())
        self.assertEqual(id3_tree.root.children["R"].feature, "temp")
        self.assertEqual(id3_tree.root.children["R"].children["M"].feature, "wind")
        # Classify all the training data
        for row_index in df.index:
            prediction = id3_tree.classify_example(df.loc[row_index])
            self.assertEqual(prediction, df.loc[row_index]["class"])

    def test_find_best_node_to_prune(self):
        # Training set
        cols = ["example", "outlook", "temp", "humidity", "wind", "class"]
        data = [
            [1,  "S", "H", "H", "F", "N"], #0
            [2,  "S", "H", "H", "T", "N"], #1
            [3,  "O", "H", "H", "F", "P"], #2
            [4,  "R", "M", "H", "F", "P"], #0
            [5,  "R", "C", "N", "F", "N"], #1
            [6,  "R", "C", "N", "T", "P"], #2
            [7,  "O", "C", "N", "T", "P"], #
            [8,  "S", "M", "H", "F", "N"], #0
            [9,  "S", "C", "N", "F", "P"], #1
            [10, "R", "M", "N", "F", "P"], #2
            [11, "S", "M", "N", "T", "P"], #0
            [12, "O", "M", "H", "T", "P"], #1
            [13, "O", "H", "N", "F", "P"], #2
            [14, "R", "M", "H", "T", "N"], #0
        ]
        df = pd.DataFrame(data, columns=cols)
        # Validation set
        validation_cols = ["example", "outlook", "temp", "humidity", "wind", "class"]
        # These are items 12 and 13 with classes switched from P to N so performance accuracy on the validation set
        # should be 0.0
        validation_data = [
            [12, "O", "M", "H", "T", "N"], #1
            [13, "O", "H", "N", "F", "N"], #2
        ]
        validation_df = pd.DataFrame(validation_data, columns=validation_cols)
        # Build tree
        id3_tree = ID3ClassificationTree(df.drop("example", axis=1), "class")
        id3_tree.build_tree(id3_tree.root)
        id3_tree.prune_tree(validation_df)
        # Initial percision accuracy is 0.0 so all children should be pruned from the tree
        self.assertTrue(len(id3_tree.node_store) == 1)

        