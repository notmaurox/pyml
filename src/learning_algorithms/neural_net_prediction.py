import logging
import sys
import random
import os
import pandas as pd
import numpy as np
import time

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

class NetworkLayer(object):

    # Class constuctor that stores a list of weights where the number of items in the list are the number of neurons in the
    # layer and each item in the list is the weight vector for that neuron. Also contains other attributes for the
    # tracking of values during back propogation.
    def __init__(self, num_weights: int, num_neurons: int, do_regression: bool, linear_comb=False):
        self.neurons = num_neurons
        self.num_weights = num_weights
        self.do_regression = do_regression
        self.weights = [[random.random() for _ in range(num_weights)] for _ in range(num_neurons)]
        self.weight_adj = [[[] for _ in range(num_weights)] for _ in range(num_neurons)]
        self.last_output = [0 for _ in range(num_neurons)]
        self.last_input = None
        self.deltas = [0 for _ in range(num_weights)]
        self.inputs = [0 for _ in range(num_neurons)]
        self.linear_comb = linear_comb
    
    # Used to print the weights of all the neurons in a layer
    def print(self):
        for neuron_index in range(len(self.weights)):
            print(f"Neuron {neuron_index} weights: {[str(weight)[:14] for weight in self.weights[neuron_index]]}")

    # Apply batch update to the weights in the layer by looking at all the proposed weight adjustments stored in the
    # self.weight_adj list
    def batch_update(self):
        for neuron_index in range(self.neurons):
            for weight_index in range(len(self.weights[neuron_index])):
                # print(self.last_input)
                # print(self.weight_adj[neuron_index][weight_index])
                self.weights[neuron_index][weight_index] -= (
                    sum(self.weight_adj[neuron_index][weight_index])
                )
        self.weight_adj = [[[] for _ in range(self.num_weights)] for _ in range(self.neurons)]
        LOG.info(f"Performed batch update of weights")

    # Reset class attributes that were used for a single round of backpropogation
    def reset(self):
        self.last_input = None
        self.last_output = [0 for _ in range(self.neurons)]
        self.delta = [0 for _ in range(self.neurons)]

    # Apply each neuron in the layer to an input vector, store the last input and output values, and return output
    # of the layer
    def apply_layer(self, example: pd.DataFrame):
        example = np.array(example)
        LOG.debug(f"Networking: {example}")
        self.last_input = example # track the last input for back propogation later
        # This is used to designate the output layer of a regression neural netowrk that only contains a single neuron
        # perofrming a linear combination of the outputs of the last hidden layer
        if self.linear_comb:
            self.last_output = [self.apply_neuron_linear(example, self.weights[0])]
            return self.last_output[0]
        layer_output = []
        # If not the special case of a regression output layer, apply each neuron to the input vector and return the
        # results of each as an output vector.
        for neuron_index in range(self.neurons):
            layer_output.append(
                self.apply_neuron_sigmoid(example, self.weights[neuron_index])
            )
        # Track the last output for backpropogation later
        self.last_output = layer_output
        LOG.debug(f"Outputting: {layer_output}")
        return layer_output

    # Apply the weights of the neuron to an input as a linear combination
    def apply_neuron_linear(self, example: pd.DataFrame, w: List[float]):
        ret = (np.dot(w, example)) # Linear activation
        return ret

    # Apply the weights of the neuron to an input with sigmoid activation applied
    def apply_neuron_sigmoid(self, example: pd.DataFrame, w: List[float]):
        LOG.debug(f"Sigmoid activation of {example} on neuron with weights...")
        LOG.debug(f"Neuron weight vector... {w}")
        ret = 1.0 / (1.0 + np.exp(-1.0 * (np.dot(w, example)))) # Sigmoid activation
        LOG.debug(f"Resulted in {ret}")
        return ret
    
    # Using the last output of the layer, calculate the transfer derivative for each node in the layer and return as
    # a vector...
    def transfer_derivatives(self):
        return [(output * (1.0 - output)) for output in self.last_output]

class NeuralNetwork(object):

    # Class constuctor that takes a data frame consisting of examples used for training, a string designating the class
    # label, integers for the number of hidden layer, number of neurons per hidden layer, a boolean setting if
    # regression or classification will occur, and a learning rate...
    def __init__(
        self, data: pd.DataFrame, class_col: str,
        num_hidden_layers: int, num_hidden_layer_neurons: int, do_regression: bool, learning_rate: float
    ):
        LOG.info(f"Initialized Neural")
        self.do_regression = do_regression
        self.class_col = class_col
        self.labels = data[class_col]
        self.data = data.drop(class_col, axis=1)
        self.layers = []
        self.learning_rate = learning_rate
        # constuct num_hidden_layers many hidden layers...
        self.layers.append(NetworkLayer(len(self.data.columns), num_hidden_layer_neurons, do_regression))
        for _ in range(num_hidden_layers-1):
            self.layers.append(NetworkLayer(num_hidden_layer_neurons, num_hidden_layer_neurons, do_regression))
        # construct output layer...
        if not do_regression:
            self.layers.append(NetworkLayer(num_hidden_layer_neurons, len(self.labels.unique()), do_regression))
            self.class_names = self.labels.unique()
        else:
            self.layers.append(NetworkLayer(num_hidden_layer_neurons, 1, do_regression, linear_comb=True))
        LOG.info(f"Learning rate set to: {self.learning_rate}")
        LOG.info(f"Inputs have {len(self.data.columns)} features")
        LOG.info(f"Initialized Neural Network with {len(self.layers)} layers")
        LOG.info(f"Layers had neuron counts...")
        for layer_index in range(len(self.layers)):
            LOG.info(f"    layer {str(layer_index+1)} has {str(self.layers[layer_index].neurons)} neurons each with {str(len(self.layers[layer_index].weights[0]))} weights")

    # Pass an example through each layer of the network and return the result of the output layer
    def network_example(self, example: pd.DataFrame, do_print=False):
        networked_example = np.array(example)
        if do_print:
            print(networked_example)
        for layer in self.layers:
            networked_example = layer.apply_layer(networked_example)
            if do_print:
                print(networked_example)
        return networked_example

    # Given the expected output of an example that was just networked thouth the neural net, backpropogate error
    # The class scores dictionary must be provided when doing classificaiton where the keys are a class name and the
    # values are the adjusted probability of that class from applying softmax. The apply_change flag is used to
    # track proposed weight changes but not actually apply them.
    def back_propogate_error_from_output_expectation(self, expected_output, class_scores=None, apply_change=False):
        # Determine delta of output layer...
        LOG.debug("Backpropogating error...")
        output_layer = self.layers[-1]
        if self.do_regression: # Derivative of MSE
            # print(output_layer.last_output[0], expected_output)
            deltas = np.array([-(expected_output[i]-output_layer.last_output[i]) for i in range(len(expected_output))])
            # exit()
        else: # Derivative of softmax...
            deltas = []
            index = 0
            for class_name in self.class_names:
                delta = 0
                # dCrossEntropy/dSoftmaxOut
                if class_name == expected_output:
                    delta += (-1/class_scores[class_name])
                else:
                    delta += (1/(1-class_scores[class_name]))
                # Multiply by dSoftmaxOut/dNeuronOutput
                delta = delta * (
                    (output_layer.last_output[index] * (sum(output_layer.last_output)-output_layer.last_output[index]))
                    / (sum(output_layer.last_output)**2)
                )
                deltas.append(delta)
                index += 1
            # The last real layer in the nextwork prior to softmax calculation applied sigmoid function so multiply
            # by transfer derivative to generate deltas contextualized to that layer...
            deltas = np.array(deltas) * np.array(output_layer.transfer_derivatives())
            LOG.debug(f"Output layer deltas: {deltas}")
        output_layer.deltas = deltas
        # Back proprogate from output layer...
        for i in reversed(range(len(self.layers)-1)):
            curr_layer = self.layers[i]
            errors = []
            for curr_layer_neuron_index in range(len(curr_layer.weights)):
                curr_layer_neuron_error = 0
                # For each neuron in previous layer, scale that neurons delta by the weight connecting to neuron
                # in current layer and sum over all neurons in previous layer...
                for prev_layer_neuron_index in range(len(self.layers[i+1].weights)):
                    # Taking dErr/dOut * dOut/dIn * dIn/dWeight
                    #       |-------- delta -----| |----weight of edge--|
                    curr_layer_neuron_error += (
                        self.layers[i+1].deltas[prev_layer_neuron_index]
                        * self.layers[i+1].weights[prev_layer_neuron_index][curr_layer_neuron_index]
                    )
                errors.append(curr_layer_neuron_error)
            curr_layer.deltas = np.array(errors) * np.array(curr_layer.transfer_derivatives())
            LOG.debug(f"Hidden layer layer deltas: {curr_layer.deltas}")
        # Iterate though layers applying deltas to calculate weight updates...
        for layer_index in range(len(self.layers)):
            curr_layer = self.layers[layer_index]
            LOG.debug(f"Layer weights prior to update: {curr_layer.weights}")
            for neuron_index in range(len(curr_layer.weights)):
                for weight_index in range(len(curr_layer.weights[neuron_index])):
                    # print(layer_index, neuron_index, weight_index)
                    weight_adj = (
                        self.learning_rate * curr_layer.deltas[neuron_index] * curr_layer.last_input[weight_index]
                    )
                    if apply_change:
                        curr_layer.weights[neuron_index][weight_index] -= weight_adj
                    curr_layer.weight_adj[neuron_index][weight_index].append(weight_adj)
            LOG.debug(f"Layer weights post update: {curr_layer.weights}")

    # Take the output of a networked example that is being classified, compute softmax calculation, return best class
    # and a dictionary where the key is the name of the class and the value is the softmax adjusted probability of the class
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

    # Go through each layer and apply updates that have been set from previous back_propogate_error_from_output_expectation
    # call where apply_change had been set to false...
    def apply_updates(self):
        for layer in self.layers:
            layer.batch_update()
            layer.reset()

    # Given a max number of iterations, shuffle the training examples and iterate through them to train the network. 
    # As currently implemented this applies stochastic gradient decent where each example is used to update the weights
    # after it has been propogated through the network.
    def train_network(self, iterations: int):
        mean_squared_errors = []
        # self.print()
        for iteration_index in range(iterations):
            iteration_error = 0
            examples_trained = 0
            # Iterate through the training examples in random order...
            row_indicies = list(self.data.index)
            random.shuffle(row_indicies)
            for example_index in row_indicies:
                networked_example = self.network_example(self.data.loc[example_index])
                expected_output = self.labels.loc[example_index]
                # Calculate error
                if self.do_regression:
                    prediction = networked_example
                    iteration_error += (expected_output - prediction)**2
                    # Backpropogate error from this example
                    self.back_propogate_error_from_output_expectation([expected_output], None, apply_change=True)
                else:
                    # Apply softmaxx...
                    best_class, class_scores = self.apply_softmax(networked_example)
                    if expected_output != best_class:
                        iteration_error += 1
                    # Backpropogate error from this example passing class scores for derivative calculation...
                    self.back_propogate_error_from_output_expectation(expected_output, class_scores, apply_change=True)
                examples_trained += 1
            # self.apply_updates()
            pad = 4-len(str(iteration_index))
            if self.do_regression:
                mse = ( iteration_error / len(self.data))
                LOG.info(f"Iteration {' '*pad}{iteration_index} MSE - {mse}")
                if len(mean_squared_errors) > 20 and mean(mean_squared_errors[:-20]) < mse:
                        LOG.info(f"Iteration MSE increased from recent average... exiting...")
                        return iteration_index
                mean_squared_errors.append(mse)
            else:
                accuracy = round(((len(self.data) - iteration_error) / len(self.data)), 4)
                LOG.info(f"Iteration {' '*pad}{iteration_index} accuracy - {accuracy} ({iteration_error} missclassified)")
        # self.print()
        return iteration_index

    # Given a data frame of test examples, iterate though them classifying each. Add predictions as a column labeled with
    # "prediction" to the dataframe
    def classify_examples(self, examples: pd.DataFrame):
        predicted_classes = []
        data = examples.drop(self.class_col, axis=1)
        for row_index in data.index:
            networked_example = self.network_example(data.loc[row_index])
            # print(examples.loc[row_index], networked_example)
            if self.do_regression:
                predicted_label = networked_example
            else:
                predicted_label, _ = self.apply_softmax(networked_example)
            predicted_classes.append(predicted_label)
        examples["prediction"] = pd.Series(predicted_classes, index=examples.index)
        return examples

    # Print the network by calling each layers print method
    def print(self):
        for layer_index in range(len(self.layers)):
            print(f"Layer {layer_index} info...")
            self.layers[layer_index].print()

    # Given a NetworkLayer from an autoencoder, replace the first layer of the network with it and adjust number of 
    # weights in the following layer...
    def apply_autoencoder_layer(self, encoder_layer: NetworkLayer):
        self.layers[0] = encoder_layer
        num_hidden_neurons = len(self.layers[1].weights)
        self.layers[1] = NetworkLayer(len(encoder_layer.weights), num_hidden_neurons, self.do_regression)


class Autoencoder(NeuralNetwork):
    # An extension of a NeuralNetwork with limited functionality. It only cares to learn the input date by
    # encoding it in one layer and decoding it in the next. all methods are the same except train network as the 
    # expected output is the same as int input...
    def __init__(self, data: pd.DataFrame, class_col: str, num_hidden_layer_neurons: int, learning_rate: float):
        LOG.info(f"Initialized AutoEncoder")
        self.do_regression = True
        self.data = data.drop(class_col, axis=1)
        self.layers = []
        self.learning_rate = learning_rate
        # Encoding layer...
        self.layers.append(
            NetworkLayer(len(self.data.columns), num_hidden_layer_neurons, True)
        )
        # Decoding layer...
        self.layers.append(
            NetworkLayer(num_hidden_layer_neurons, len(self.data.columns), True)
        )
        LOG.info(f"Learning rate set to: {self.learning_rate}")
        LOG.info(f"Inputs have {len(self.data.columns)} features")
        LOG.info(f"Initialized Neural Network with {len(self.layers)} layers")
        LOG.info(f"Layers had neuron counts...")
        for layer_index in range(len(self.layers)):
            LOG.info(f"    layer {str(layer_index+1)} has {str(self.layers[layer_index].neurons)} neurons each with {str(len(self.layers[layer_index].weights[0]))} weights")

    # Iterate through the training data in random order iterations number of times encoding and decoding each example.
    # 
    def train_network(self, iterations: int):
        prev_mse = float('inf')
        mean_squared_errors = []
        # self.print()
        for iteration_index in range(iterations):
            iteration_error = 0
            examples_trained = 0
            # Iterate in random order
            row_indicies = list(self.data.index)
            random.shuffle(row_indicies)
            for example_index in row_indicies:
                networked_example = self.network_example(self.data.loc[example_index])
                # Notice that expected output = input initially provided to the network
                expected_output = self.data.loc[example_index]
                # Calculate error using MSE
                iteration_error += sum((expected_output - networked_example)**2)
                self.back_propogate_error_from_output_expectation(expected_output, None, apply_change=True)
                examples_trained += 1
            # self.apply_updates()
            pad = 4-len(str(iteration_index))
            mse = ( iteration_error / len(self.data))
            LOG.info(f"Iteration {' '*pad}{iteration_index} MSE - {mse}")
            if len(mean_squared_errors) > 6:
                if mean(mean_squared_errors[:-5]) < mse:
                    LOG.info(f"Iteration MSE increased from recent average... exiting...")
                    return
            LOG.info(f"Auto encoder weights... ")
            self.print()






        


