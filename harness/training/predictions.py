import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from utils import dataframe_helper


def classify(predictions, threshold):
    """Turns raw predictions into actual classification. Only 1/0 for now."""
    return np.where(predictions > threshold, 1, 0)


class Predictions:
    """Class to store and handle prediction results."""
    DEFAULT_THRESHOLD = 0.5
    ACCURACY_TRUE_POSITIVE = "tp"
    ACCURACY_TRUE_NEGATIVE = "tn"
    ACCURACY_FALSE_POSITIVE = "fp"
    ACCURACY_FALSE_NEGATIVE = "fn"
    POSITIVE_CLASS = 1

    TRUTH_KEY = "truth"
    PREDICTIONS_KEY = "prediction"
    RAW_PREDICTIONS_LEY = "raw_prediction"

    classification_threshold = 0
    raw_predictions = None
    predictions = None
    expected_results = None
    tf_pn_by_sample = None
    total_true_positives = 0
    total_true_negatives = 0
    total_false_positives = 0
    total_false_negatives = 0

    def __init__(self, classification_threshold=DEFAULT_THRESHOLD):
        self.classification_threshold = classification_threshold

    def store_expected_results(self, expected_output):
        """Stores expected results and confusion matrix."""
        self.expected_results = expected_output
        if self.tf_pn_by_sample is None:
            self._calculate_true_false_positives_negatives()

    def get_expected_results(self):
        """Returns the ground truth classification."""
        return self.expected_results

    def store_raw_predictions(self, raw_predictions):
        """Stores raw and classified predicitons."""
        self.raw_predictions = raw_predictions

    def classify_raw_predictions(self):
        """Generates classified predictions based on the raw ones."""
        if self.raw_predictions is not None:
            self.store_predictions(classify(self.raw_predictions, self.classification_threshold))

    def store_predictions(self, predictions):
        """Stores already classified predictions."""
        self.predictions = predictions
        if self.tf_pn_by_sample is None:
            self._calculate_true_false_positives_negatives()

    def get_raw_predictions(self):
        """Returns the raw predictions."""
        return self.raw_predictions

    def get_predictions(self):
        """Returns the classified predictions."""
        return self.predictions

    def create_slice(self, starting_idx, size):
        """Creates a new TrainingResults object with a slice of the results in this one."""
        sliced_training_results = Predictions(self.classification_threshold)
        sliced_training_results.store_expected_results(self.get_expected_results()[starting_idx:starting_idx + size])
        sliced_training_results.store_predictions(self.get_predictions()[starting_idx:starting_idx + size])
        return sliced_training_results

    def _calculate_true_false_positives_negatives(self):
        """Calculates confusion matrix, and for each sample if it was a true/false positive/negative."""
        if self.expected_results is not None and self.predictions is not None:
            conf_matrix = confusion_matrix(self.expected_results, self.predictions)
            self.total_true_negatives, self.total_false_positives, self.total_false_negatives, self.total_true_positives \
                = conf_matrix.ravel()
            print(f"TN: {self.total_true_negatives}, TP: {self.total_true_positives}, "
                  f"FN: {self.total_false_negatives}, FP: {self.total_false_positives}")

            self.tf_pn_by_sample = []
            for idx, truth in enumerate(self.expected_results):
                if truth == self.predictions[idx]:
                    if truth == self.POSITIVE_CLASS:
                        self.tf_pn_by_sample.append(self.ACCURACY_TRUE_POSITIVE)
                    else:
                        self.tf_pn_by_sample.append(self.ACCURACY_TRUE_NEGATIVE)
                else:
                    if truth == self.POSITIVE_CLASS:
                        self.tf_pn_by_sample.append(self.ACCURACY_FALSE_POSITIVE)
                    else:
                        self.tf_pn_by_sample.append(self.ACCURACY_FALSE_NEGATIVE)

    def get_accuracy(self):
        """Calculates the accuracy and returns it"""
        return (self.total_true_negatives + self.total_true_positives) / \
               (self.total_true_negatives + self.total_true_positives + self.total_false_positives + self.total_false_negatives)

    def as_dataframe(self, dataframe=None):
        """Returns a dataframe with this predictions object, adding to existing argument if received."""
        if dataframe is None:
            dataframe = pd.DataFrame()

        dataframe[self.TRUTH_KEY] = self.get_expected_results()
        dataframe[self.PREDICTIONS_KEY] = self.get_predictions()
        dataframe[self.RAW_PREDICTIONS_LEY] = self.get_raw_predictions()
        return dataframe

    def save_to_file(self, output_filename, base_df=None):
        """Saves this prediction object to file"""
        full_df = self.as_dataframe(base_df)
        dataframe_helper.save_dataframe_to_file(full_df, output_filename)

    def load_from_file(self, predictions_filename):
        """Loads predictions info from a file."""
        dataset_df = dataframe_helper.load_dataframe_from_file(predictions_filename)
        self.store_expected_results(np.array(dataset_df[self.TRUTH_KEY]))
        self.store_raw_predictions(np.array(dataset_df[self.RAW_PREDICTIONS_LEY]))
        self.store_predictions(np.array(dataset_df[self.PREDICTIONS_KEY]))
