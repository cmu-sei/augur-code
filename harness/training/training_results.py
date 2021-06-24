import numpy as np
from sklearn.metrics import confusion_matrix


def classify(predictions, threshold):
    """Turns raw predictions into actual classification. Only 1/0 for now."""
    return np.where(predictions > threshold, 1, 0)


class TrainingResults:
    """Class to store and handle prediction results."""
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
        """Returns the classified predictions."""
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

    def get_predictions(self):
        """Returns the classified predictions."""
        return self.predictions

    def create_slice(self, starting_idx, size):
        """Creates a new TrainingResults object with a slice of the results in this one."""
        sliced_training_results = TrainingResults(self.classification_threshold)
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
            return tn, fp, fn, tp
        else:
            raise Exception("Confusion matrix has not been computed yet.")

    def get_accuracy(self):
        """Calculates the accuracy and returns it"""
        tn, fp, fn, tp = self._get_true_false_positives_negatives()
        return (tn + tp) / (tn + tn + fp + fn)
