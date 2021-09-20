import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from utils import dataframe_helper


def classify(predictions, threshold):
    """Turns raw predictions into actual classification. Only 1/0 for now."""
    return np.where(predictions > threshold, 1, 0)


class Predictions:
    """Class to store and handle prediction results."""
    TRUTH_KEY = "truth"
    PREDICTIONS_KEY = "prediction"
    RAW_PREDICTIONS_LEY = "raw_prediction"

    classification_threshold = 0.5
    raw_predictions = None
    predictions = None
    expected_results = None
    conf_matrix = None

    def __init__(self, classification_threshold):
        self.classification_threshold = classification_threshold

    def store_expected_results(self, expected_output):
        """Stores expected results and confusion matrix."""
        self.expected_results = expected_output
        if self.conf_matrix is None:
            self._calculate_confusion_matrix()

    def get_expected_results(self):
        """Returns the ground truth classification."""
        return self.expected_results

    def store_raw_predictions(self, raw_predictions):
        """Stores raw and classified predicitons."""
        self.raw_predictions = raw_predictions
        self.predictions = classify(self.raw_predictions, self.classification_threshold)
        if self.conf_matrix is None:
            self._calculate_confusion_matrix()

    def store_predictions(self, predictions):
        """Stores already classified predictions."""
        self.predictions = predictions
        if self.conf_matrix is None:
            self._calculate_confusion_matrix()

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

    def _calculate_confusion_matrix(self):
        """Calculates the confusion matrix."""
        if self.expected_results is not None and self.predictions is not None:
            self.conf_matrix = confusion_matrix(self.expected_results, self.predictions)

    def _get_true_false_positives_negatives(self):
        """Given a set of predictions, returns the true/false positives/negatives as a dict."""
        if self.conf_matrix is not None:
            tn, fp, fn, tp = self.conf_matrix.ravel()
            print(f"TN: {tn}, TP: {tp}, FN: {fn}, FP: {fp}")
            return tn, fp, fn, tp
        else:
            raise Exception("Confusion matrix has not been computed yet.")

    def get_accuracy(self):
        """Calculates the accuracy and returns it"""
        tn, fp, fn, tp = self._get_true_false_positives_negatives()
        return (tn + tp) / (tn + tp + fp + fn)

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

        self._calculate_confusion_matrix()
        self.get_accuracy()
