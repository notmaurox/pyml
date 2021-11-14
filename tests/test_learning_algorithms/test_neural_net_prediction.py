import unittest
import sys
import os
import pathlib
import statistics
import logging

import pandas as pd

PATH_TO_SRC_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), "../", "../", "src/")
sys.path.insert(0, PATH_TO_SRC_DIR)

from learning_algorithms.neural_net_prediction import NetworkLayer, NeuralNetwork
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

class TestNeuralNetwork(unittest.TestCase):

    # Working example of: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    # No longer functional....
    @unittest.skip
    def test_back_prop(self):
        cols = ["f1", "f2", "class"]
        data = [
            [0.05, 0.10, "c1"], #0
            [0.05, 0.10, "c1"], #1
        ]
        df = pd.DataFrame(data, columns=cols)
        nn = NeuralNetwork(df, "class", 2, 1, do_regression=False)
        # make l1 from example
        l1 = NetworkLayer(2, 2, do_regression=False)
        l1.weights = [
            [.15, .20],
            [.25, .30]
        ]
        l1.biases = [0.35, 0.35]
        # make l2 from example
        l2 = NetworkLayer(2, 2, do_regression=False)
        l2.weights = [
            [.40, .45],
            [.50, .55]
        ]
        l2.biases = [0.60, 0.60]
        nn.layers = [l1, l2]
        networked_example = [0.05, 0.10]
        for layer in nn.layers:
            networked_example = layer.apply_layer(networked_example)
        # 0.75136... 0.77292...
        self.assertEqual(networked_example, [0.7513650695523157, 0.7729284653214625])
        nn.back_propogate_error_from_output_expectation([0.01, 0.99])

        self.assertEqual(nn.layers[0].weights[0], [0.1497807161327628, 0.19956143226552567])
        self.assertEqual(nn.layers[0].weights[1], [0.24975114363236958, 0.29950228726473915])
        self.assertEqual(nn.layers[1].weights[0], [0.35891647971788465, 0.4086661860762334])
        self.assertEqual(nn.layers[1].weights[1], [0.5113012702387375, 0.5613701211079891])