import logging
import sys
import random
import pandas as pd
import numpy as np

from typing import List
from math import log
from statistics import mean

# Logging stuff
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
LOG.addHandler(handler)

LEARNING_RATE = 0.25

class NetworkLayer(object):
    
    def __init__(self, num_weights: int, num_neurons: int, do_regression: bool):
        self.neurons = num_neurons
        self.num_weights = num_weights
        self.do_regression = do_regression
        self.weights = [[random.random() for _ in range(num_weights)] for _ in range(num_neurons)]
        self.weight_adj = [[[] for _ in range(num_weights)] for _ in range(num_neurons)]
        self.biases = [0 for _ in range(num_neurons)]
        self.biases_adj = [[] for _ in range(num_neurons)]
        self.last_output = [0 for _ in range(num_neurons)]
        self.last_input = None
        self.deltas = [0 for _ in range(num_weights)]
        self.inputs = [0 for _ in range(num_neurons)]
        
    def print_weights(self):
        print("Neuron weights...")
        for row in self.weights:
            print(row)

    def batch_update(self):
        for neuron_index in range(self.neurons):
            for weight_index in range(len(self.weights[0])):
                self.weights[neuron_index][weight_index] -= (
                    mean(self.weight_adj[neuron_index][weight_index])
                )
            self.biases[neuron_index] -= (
                mean(self.biases_adj[neuron_index])
            )
        self.weight_adj = [[[] for _ in range(self.num_weights)] for _ in range(self.neurons)]
        self.biases_adj = [[] for _ in range(self.num_weights)]

    def reset(self):
        self.last_output = [0 for _ in range(self.neurons)]
        self.delta = [0 for _ in range(self.neurons)]

    def apply_layer(self, example: pd.DataFrame):
        example = np.array(example)
        self.last_input = example
        # This layer is a single output node for regression...
        if self.do_regression and self.neurons == 1:
            self.last_output[0] = np.dot(self.weights[0], example) + self.biases[0]
            return self.last_output[0]
        layer_output = []
        for neuron_index in range(self.neurons):
            layer_output.append(
                self.apply_neuron(example, self.weights[neuron_index], self.biases[neuron_index])
            )
        self.last_output = layer_output
        return layer_output

    def apply_neuron(self, example: pd.DataFrame, w: List[float], b: float):
        return 1.0 / (1.0 + np.exp(-1.0 * (np.dot(w, example) + b))) # Sigmoid activation
    
    def transfer_derivatives(self):
        return [(output * (1.0 - output)) for output in self.last_output]

class NeuralNetwork(object):

    def __init__(
        self, data: pd.DataFrame, class_col: str,
        num_hidden_layers: int, num_hidden_layer_neurons: int, do_regression: bool
    ):
        self.do_regression = do_regression
        self.class_col = class_col
        self.labels = data[class_col]
        self.data = data.drop(class_col, axis=1)
        self.layers = []
        # constuct num_hidden_layers many hidden layers...
        self.layers.append(NetworkLayer(len(self.data.columns), num_hidden_layer_neurons, do_regression))
        for _ in range(num_hidden_layers-1):
            self.layers.append(NetworkLayer(num_hidden_layer_neurons, num_hidden_layer_neurons, do_regression))
        # construct output layer...
        if not do_regression:
            self.layers.append(NetworkLayer(num_hidden_layer_neurons, len(self.labels.unique()), do_regression))
            self.class_names = self.labels.unique()
        else:
            self.layers.append(NetworkLayer(num_hidden_layer_neurons, 1, do_regression))

    def network_example(self, example: pd.DataFrame):
        networked_example = np.array(example)
        for layer in self.layers:
            networked_example = layer.apply_layer(networked_example)
        return networked_example

    def classify_networked_example(self, networked_example):
        if self.do_regression:
            # Do a linear combination of output layer...
            return networked_example, None
        else:
            # Do soft max of output layer...
            class_scores = {}
            for class_index in range(len(self.class_names)):
                class_scores[self.class_names[class_index]] = np.exp(networked_example[class_index])
            denom = sum(score for score in class_scores.values())
            best_class, best_class_score = 0, 0
            for class_name in class_scores.keys():
                class_scores[class_name] = (class_scores[class_name] / denom)
                if class_scores[class_name] > best_class_score:
                    best_class, best_class_score = class_name, class_scores[class_name]
            return best_class, class_scores

    def back_propogate_error_from_output_expectation(self, expected_output, class_scores=None, apply_change=False):
        # Determine delta of output layer...
        output_layer = self.layers[-1]
        if self.do_regression: # Derivative of MSE
            deltas = [-(expected_output[i]-output_layer.last_output[i]) for i in range(len(output_layer.last_output))]
        else: # Derivative of softmax...
            deltas = []
            for class_name in self.class_names:
                if class_name == expected_output:
                    deltas.append(class_scores[class_name]-1)
                else:
                    deltas.append(class_scores[class_name]-0)
        deltas = np.array(deltas) * np.array(output_layer.transfer_derivatives())
        output_layer.deltas = deltas
        # Back proprogate...
        for i in reversed(range(len(self.layers)-1)):
            curr_layer = self.layers[i]
            errors = []
            for curr_layer_neuron_index in range(curr_layer.neurons):
                curr_layer_neuron_error = 0
                # For each neuron in previous layer, scale that neurons delta by the weight connecting to neuron
                # in current layer and sum over all neurons in previous layer...
                for prev_layer_neuron_index in range(self.layers[i+1].neurons):
                    # Taking dErr/dOut * dOut/dIn * dIn/dWeight
                    #       |-------- delta -----| |----weight of edge--|
                    curr_layer_neuron_error += (
                        self.layers[i+1].deltas[prev_layer_neuron_index]
                        * self.layers[i+1].weights[prev_layer_neuron_index][curr_layer_neuron_index]
                    )
                errors.append(curr_layer_neuron_error)
            curr_layer.deltas = np.array(errors) * np.array(curr_layer.transfer_derivatives())
        # update output neurons weights...
        for neuron_index in range(output_layer.neurons):
            for weight_index in range(len(output_layer.weights[0])):
                weight_adj = (
                    LEARNING_RATE * output_layer.deltas[neuron_index] * output_layer.last_input[weight_index]
                )
                if apply_change:
                    output_layer.weights[neuron_index][weight_index] -= weight_adj
                output_layer.weight_adj[neuron_index][weight_index].append(weight_adj)
            bias_adj = (
                LEARNING_RATE * output_layer.deltas[neuron_index]
            )
            if apply_change:
                output_layer.biases[neuron_index] -= bias_adj
            output_layer.biases_adj[neuron_index].append(bias_adj)
        # update hidden neurons weights...
        for layer_index in reversed(range(len(self.layers)-1)):
            curr_layer = self.layers[layer_index]
            for neuron_index in range(curr_layer.neurons):
                for weight_index in range(len(curr_layer.weights[0])):
                    weight_adj = (
                        LEARNING_RATE * curr_layer.deltas[neuron_index] * curr_layer.last_input[weight_index]
                    )
                    if apply_change:
                        curr_layer.weights[neuron_index][weight_index] -= weight_adj
                    curr_layer.weight_adj[neuron_index][weight_index].append(weight_adj)
                bias_adj = (
                    LEARNING_RATE * curr_layer.deltas[neuron_index]
                )
                if apply_change:
                    curr_layer.biases[neuron_index] -= bias_adj
                curr_layer.biases_adj[neuron_index].append(bias_adj)

    def apply_softmax(self, networked_example):
        # Do soft max of output layer...
        class_scores = {}
        for class_index in range(len(self.class_names)):
            class_scores[self.class_names[class_index]] = np.exp(networked_example[class_index])
        denom = sum(score for score in class_scores.values())
        best_class, best_class_score = 0, 0
        for class_name in class_scores.keys():
            class_scores[class_name] = (class_scores[class_name] / denom)
            if class_scores[class_name] > best_class_score:
                best_class, best_class_score = class_name, class_scores[class_name]
        return best_class, class_scores

    def apply_updates(self):
        for layer in self.layers:
            layer.batch_update()

    def train_network(self, iterations: int):
        for iteration_index in range(iterations):
            iteration_error = 0
            examples_trained = 0
            for example_index in self.data.index:
                networked_example = self.network_example(self.data.loc[example_index])
                expected_output = self.labels.loc[example_index]
                # Calculate error
                if self.do_regression:
                    pass
                    # iteration_error += (expected_output - prediction)**2
                    # self.back_propogate_deltas(expected_output, None, False)
                else:
                    # Apply softmaxx...
                    best_class, class_scores = self.apply_softmax(networked_example)
                    if expected_output != best_class:
                        iteration_error += 1
                    self.back_propogate_error_from_output_expectation(expected_output, class_scores, apply_change=True)
                examples_trained += 1
                # exit()
            # self.apply_updates()
            if self.do_regression:
                mse = ( iteration_error / len(self.data))
                LOG.info(f"Iteration MSE: {mse}")
            else:
                accuracy = round(((len(self.data) - iteration_error) / len(self.data)), 4)
                LOG.info(f"Iteration {iteration_index}: Iteration accuracy: {accuracy} ({iteration_error} missclassified)")

    def classify_examples(self, examples):
        predicted_classes = []
        data = examples.drop(self.class_col, axis=1)
        for row_index in data.index:
            networked_example = self.network_example(data.loc[row_index])
            if self.do_regression:
                pass
            else:
                predicted_label, _ = self.apply_softmax(networked_example)
            predicted_classes.append(predicted_label)
        examples["prediction"] = pd.Series(predicted_classes, index=examples.index)
        return examples

    def print(self):
        for layer in self.layers:
            print(layer.weights)





        


