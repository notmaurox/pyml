import logging
import sys
import random
import pandas as pd
import numpy as np


# Logging stuff
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
LOG.addHandler(handler)

class LinearRegressor(object):
    
    # Class constructor accepts a data frame with the examples to be used for learning and a string designating the class
    def __init__(self, data: pd.DataFrame, class_col: str):
        self.class_col = class_col
        self.labels = data[class_col]
        self.data = data.drop(class_col, axis=1)
        # Initialize weight vector...
        self.wight_vector = np.random.uniform(low=-0.01, high=0.01, size=(len(self.data.columns),))
        self.b = 0
        # self.learning_rate = 10**-1
        self.learning_rate = 0.80

    # Iterate though the training data in random order untill an iteration produces more missclassified examples than
    # the prior iteration.
    def learn(self, min_iterations=None):
        mse = float('inf')
        iterations = 0
        num_samples_per_iteration = int(len(self.data))
        mses = []
        while True:
            mse_sum = 0
            adjustment_vector = np.zeros(len(self.data.columns))
            adjustment_b = 0
            # Shuffle training data iteration order
            row_indicies = list(self.data.index)
            random.shuffle(row_indicies)
            for example_index in row_indicies[:num_samples_per_iteration]:
                dot_prod = (np.dot(self.wight_vector, self.data.loc[example_index]) + self.b)
                # dot_prod = (sum(
                #     self.wight_vector[i] + self.data.loc[example_index][i] for i in range(len(self.wight_vector))
                # ) + self.b)
                error = (dot_prod - self.labels.loc[example_index]) # Calculate error between predicted and actual value 
                mse_sum += error**2 # Square the difference for MSE
                # Use unsquared diference (MSE derivative) to scale example for adjustment
                adjustment_vector += (error * self.data.loc[example_index])
                adjustment_b += error
                # # Incremental update
                # self.wight_vector -= (self.learning_rate * error * self.labels.loc[example_index])
                # self.b -= (self.learning_rate * error)
            # Batch update
            self.wight_vector = self.wight_vector - (self.learning_rate / num_samples_per_iteration) * adjustment_vector
            self.b = self.b - (self.learning_rate / num_samples_per_iteration) * adjustment_b
            iterations += 1
            curr_mse = (mse_sum / num_samples_per_iteration)
            if min_iterations != None:
                # Continue for at least min iterations...
                if iterations < min_iterations:
                    mse = curr_mse
                    mses.append(mse)
                    continue
            # Once reaching a fold where mse has increased >1% since last iteration, stop learning
            if curr_mse >= mse or (mse-curr_mse) < (mse*0.01):
                LOG.info(f"Stopped learning on iteration with {curr_mse} mse from {mse} during the previous iteration")
                LOG.info(f"Learning produced iterations with the following MSEs: {mses}")
                LOG.info(f"Finished after {iterations} iterations though example points with {curr_mse} mse")
                return iterations, mses
            mse = curr_mse
            mses.append(mse)

    # Apply dot product of weight vector and example
    def classify_example(self, example: pd.DataFrame):
        return np.dot(self.wight_vector, example) + self.b

    # Iterate through a dataframe of training examples and classify each adding a column labeled "prediction"
    def classify_examples(self, examples: pd.DataFrame):
        predicted_classes = []
        data = examples.drop(self.class_col, axis=1)
        for row_index in data.index:
            predicted_label = self.classify_example(data.loc[row_index])
            predicted_classes.append(predicted_label)
        examples["prediction"] = pd.Series(predicted_classes, index=examples.index)
        return examples

