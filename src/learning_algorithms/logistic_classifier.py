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

class LogisticRegressionClassifier(object):

    # class constructor that takes a dataframe with training examples and a string designating class column...
    def __init__(self, data: pd.DataFrame, class_col: str):
        self.class_col = class_col
        self.labels = data[class_col]
        self.data = data.drop(class_col, axis=1)
        #  Hardcoded learning weight 
        self.learning_rate = 10**-3
        self.class_weight_vectors = {}
        # Each class label gets it's own weight vector for classification
        for class_name in self.labels.unique():
            self.class_weight_vectors[class_name] = np.random.uniform(low=-0.01, high=0.01, size=(len(self.data.columns),))

    # Compute dot product between weight vector and example
    def _combine(self, w, example: pd.DataFrame):
        return w[0] + np.dot(example, w)

    # Iterate through the training data in random order until an iteration is reached where the number of missclassified
    # examples increased
    def learn(self, allowed_bad_iterations=1):
        num_missclassified = len(self.data) + 1
        iterations = 0
        bad_iterations = 0
        while True:
            curr_missclassified = 0
            class_weight_updates = {}
            # Initialize store to track proposed updates from iterating through training set
            for class_name in self.labels.unique():
                class_weight_updates[class_name] = np.zeros(len(self.data.columns))
            # iterate through the data set in random order
            row_indicies = list(self.data.index)
            random.shuffle(row_indicies)
            for example_index in row_indicies:
                class_scores = {}
                # Initialize store to keep track of results of applying each classes weight vector to the example
                for class_name, class_w in self.class_weight_vectors.items():
                    class_score = self._combine(class_w, self.data.loc[example_index])
                    # Compute e^dot product for soft max calculation
                    class_scores[class_name] = np.exp(class_score)
                # Calculate sum of all class scores for softmax calculation
                denom = sum(score for score in class_scores.values())
                best_class, best_class_score = 0, 0
                # Normalize each class score by the sum for softmax calculation and select highest score...
                for class_name in class_scores.keys():
                    class_scores[class_name] = (class_scores[class_name] / denom)
                    if class_scores[class_name] > best_class_score:
                        best_class, best_class_score = class_name, class_scores[class_name]
                # Track count of misclassified exampls
                if best_class != self.labels.loc[example_index]:
                    curr_missclassified += 1
                # Apply updates to each class weight vector from resulting softmax calulcations per class
                for class_name, class_score in class_scores.items():
                    if class_name == self.labels.loc[example_index]:
                        r = 1
                    else:
                        r = 0
                    update_vector = (r - class_score)*self.data.loc[example_index]
                    class_weight_updates[class_name] += update_vector
            # BATCH update class vectors using average weight adjustment scaled byh learning rate
            for class_name in self.labels.unique():
                class_weight_updates[class_name] = (self.learning_rate * class_weight_updates[class_name])
                self.class_weight_vectors[class_name] += class_weight_updates[class_name]
                LOG.debug(f"Class {class_name} updated with vector...")
                LOG.debug((self.learning_rate * class_weight_updates[class_name]).to_list())
            iterations += 1
            # If we reach an iteration with more examples misclassified than before, exit...
            if curr_missclassified >= num_missclassified:
                bad_iterations += 1
                LOG.info(f"Had bad iteration with {curr_missclassified} misclassified from {num_missclassified} during the previous iteration")
                if bad_iterations == allowed_bad_iterations:
                    LOG.info(f"Reached allowance of {allowed_bad_iterations} bad iterations")
                    LOG.info(f"Finished after {iterations} iterations though example points with {num_missclassified} misclassifications")
                    return iterations
            num_missclassified = curr_missclassified

    # Given an test example to classify, compute dot product with each claeese weight vector, soft max normalize, and
    # return best class and associated score
    def classify_example(self, example: pd.DataFrame):
        class_scores = {}
        for class_name, class_w in self.class_weight_vectors.items():
            class_score = self._combine(class_w, example)
            class_scores[class_name] = np.exp(class_score)
        denom = sum(score for score in class_scores.values())
        best_class, best_class_score = 0, 0
        for class_name in class_scores.keys():
            class_scores[class_name] = (class_scores[class_name] / denom)
            if class_scores[class_name] > best_class_score:
                best_class, best_class_score = class_name, class_scores[class_name]
        return best_class, best_class_score
    
    # Iterate through a data frame of test examples classifying each one and adding the predicted label to the data
    # frame as a columne titled "prediction"
    def classify_examples(self, examples: pd.DataFrame):
        predicted_classes = []
        data = examples.drop(self.class_col, axis=1)
        for row_index in data.index:
            predicted_label, _ = self.classify_example(data.loc[row_index])
            predicted_classes.append(predicted_label)
        examples["prediction"] = pd.Series(predicted_classes, index=examples.index)
        return examples
