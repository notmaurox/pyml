import unittest
import sys
import os
import pathlib
import statistics
import logging

import pandas as pd

PATH_TO_SRC_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), "../", "../", "src/")
sys.path.insert(0, PATH_TO_SRC_DIR)

import learning_algorithms.ID3_decision_tree_predictor
from learning_algorithms.ID3_decision_tree_predictor import ID3ClassificationTree
from data_utils.data_loader import DataLoader
from data_utils.metrics_evaluator import MetricsEvaluator
from data_utils.data_transformer import DataTransformer

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
LOG.addHandler(handler)

class TestID3ClassificationTree(unittest.TestCase):

    # Provide sample outputs from one test set on one fold for a classification tree
    # Show a sample classification tree without pruning and with pruning.
    def test_unpruned_vs_pruned_tree(self):
        ## Test data set - Car rating
        df = DataLoader.load_car_data()
        class_col = "acceptibility" #attempt to precit acceptibility
        folds, hyperparam_set_indicies = DataTransformer.produce_k_fold_cross_validation_sets(
            df, 5, class_col, make_hyperparam_set=True, hyperparam_set_proportion=0.2
        )
        prediction_scores, node_count = [], []
        pruned_prediction_scores, pruned_node_count = [], []
        for train_indicies, test_indicies in folds:
            LOG.info("Learning on new fold...")
            LOG.info(f"Training on {len(train_indicies)} entitites and testing on {len(test_indicies)} entitites...")
            train_df = df.loc[train_indicies].copy()
            # LOG.info(f"Train set has class representation..\n{train_df[class_col].value_counts()}")
            test_df = df.loc[test_indicies].copy()
            # Build tree w/o pruning...
            LOG.info(f"Building tree...")
            id3_tree = ID3ClassificationTree(train_df, class_col)
            id3_tree.build_tree(id3_tree.root)
            # Classify w/o pruning...
            LOG.info(f"Classifying examples on tree with {len(id3_tree.node_store)} nodes")
            node_count.append(len(id3_tree.node_store))
            classified_examples = id3_tree.classify_examples(test_df)
            score = MetricsEvaluator.calculate_classification_score(classified_examples[class_col], classified_examples["prediction"])
            score = round(score, 4)
            LOG.info(f"resulting classification score: {score}")
            prediction_scores.append(score)
            # Prune...
            LOG.info(f"Pruning tree using validation set of {len(hyperparam_set_indicies)} examples")
            id3_tree.prune_tree(df.loc[hyperparam_set_indicies])
            # Classify after pruining...
            LOG.info(f"Classifying examples on tree with {len(id3_tree.node_store)} nodes")
            pruned_node_count.append(len(id3_tree.node_store))
            classified_examples = id3_tree.classify_examples(test_df)
            score = MetricsEvaluator.calculate_classification_score(classified_examples[class_col], classified_examples["prediction"])
            score = round(score, 4)
            LOG.info(f"resulting classification score: {score}")
            pruned_prediction_scores.append(score)
            break
        avrg_pred_score, avrg_node_count = statistics.fmean(prediction_scores), statistics.fmean(node_count)
        pruned_avrg_pred_score, pruned_avrg_node_count = statistics.fmean(pruned_prediction_scores), statistics.fmean(pruned_node_count)
        LOG.info(f"ID3 tree w/o pruining had on average {round(avrg_node_count, 0)} nodes with accuracy {avrg_pred_score}")
        LOG.info(f"ID3 tree w/  pruining had on average {round(pruned_avrg_node_count, 0)} nodes with accuracy {pruned_avrg_pred_score}")
        self.assertLess(pruned_avrg_node_count, avrg_node_count)

    # Demonstrate the calculation of information gain, gain ratio
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

    # Demonstrate a decision being made to prune a subtree
    def test_prune_subtree(self):
        learning_algorithms.ID3_decision_tree_predictor.handler.setLevel(logging.DEBUG)
        learning_algorithms.ID3_decision_tree_predictor.LOG.setLevel(logging.DEBUG)
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
        # Initial prediction accuracy on validation set is 0.0 so all children should be pruned from the tree
        # As classifying using the root node alone will have accuracy of 0.0
        id3_tree.prune_tree(validation_df)
        self.assertTrue(len(id3_tree.node_store) == 1)
        learning_algorithms.ID3_decision_tree_predictor.handler.setLevel(logging.INFO)
        learning_algorithms.ID3_decision_tree_predictor.LOG.setLevel(logging.INFO)

    # Demonstrate an example traversing a classification tree and a class label being assigned at the leaf...
    def test_build_tree_and_classify(self):
        learning_algorithms.ID3_decision_tree_predictor.handler.setLevel(logging.DEBUG)
        learning_algorithms.ID3_decision_tree_predictor.LOG.setLevel(logging.DEBUG)
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
            LOG.info(f'Recieved prediction {prediction} for example with actual class {df.loc[row_index]["class"]}')
            self.assertEqual(prediction, df.loc[row_index]["class"])
        learning_algorithms.ID3_decision_tree_predictor.handler.setLevel(logging.INFO)
        learning_algorithms.ID3_decision_tree_predictor.LOG.setLevel(logging.INFO)

    # Test transforming of numeric attributes with edges defined by the midpoint between class average vals...
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
        #                  example            numeric_col class
        # 0     vals_lt_5.25_gt_NA     vals_lt_7.25_gt_NA     Y
        # 1     vals_lt_5.25_gt_NA     vals_lt_7.25_gt_NA     Y
        # 2     vals_lt_5.25_gt_NA     vals_lt_7.25_gt_NA     Y
        # 3     vals_lt_5.25_gt_NA     vals_lt_7.25_gt_NA     Y
        # 4     vals_lt_5.25_gt_NA     vals_lt_7.25_gt_NA     Y
        # 5   vals_lt_9.75_gt_5.25  vals_lt_17.15_gt_7.25     N
        # 6   vals_lt_9.75_gt_5.25  vals_lt_17.15_gt_7.25     N
        # 7   vals_lt_9.75_gt_5.25  vals_lt_17.15_gt_7.25     N
        # 8   vals_lt_9.75_gt_5.25  vals_lt_17.15_gt_7.25     N
        # 9           vals_gt_9.75          vals_gt_17.15     P
        # 10          vals_gt_9.75          vals_gt_17.15     P
        # 11          vals_gt_9.75          vals_gt_17.15     P
        # 12          vals_gt_9.75          vals_gt_17.15     P
        # 13          vals_gt_9.75          vals_gt_17.15     P
        self.assertEqual(
            ["vals_lt_5.25_gt_NA", "vals_lt_9.75_gt_5.25", "vals_gt_9.75"],
            df["example"].unique().tolist()
        )
        self.assertEqual(
            ["vals_lt_7.25_gt_NA", "vals_lt_17.15_gt_7.25", "vals_gt_17.15"],
            df["numeric_col"].unique().tolist()
        )

    # test calculation of entropy - example taken from lecture slides...
    def test_calc_entropy(self):
        class_composition = {
            "pos": 9,
            "neg": 5,
        }
        total_examples = 14
        entropy = ID3ClassificationTree.calc_entropy(class_composition, total_examples)
        self.assertEqual(round(entropy, 2), 0.94)

    # Calculate feature entropy and information gain and gain ratio, example taken from lecture. 
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

    # test identifcation of best feature to split on from input dataframe...
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
        