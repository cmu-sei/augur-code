import numpy as np


class SampleGroup:
    """Represents a bounded subset of data used to generate drift."""
    id = 0
    size = 0
    accuracy = 0
    metric_name = ""
    metric_value = 0
    predictions = None
    starting_idx = 0

    def __init__(self, sample_group_id, sample_group_size):
        self.id = sample_group_id
        self.size = sample_group_size

    def set_data(self, full_predictions, starting_idx):
        """Stores the data from full predictions, for this specific sample group."""
        self.starting_idx = starting_idx
        self.predictions = full_predictions.create_slice(self.starting_idx, self.size)

    def get_expected_results(self):
        """Returns the output for this sample group, slicing it from the dataset given the start idx and size."""
        return self.predictions.get_expected_results()

    def get_number_of_samples(self):
        return self.size

    def get_predictions(self):
        """Returns the predictions for this sample group, slicing it from the dataset given the start idx and size."""
        return self.predictions.get_predictions()

    def get_correctness(self):
        """Returns the correctness of each prediction for this sample group."""
        return np.array_equal(self.get_expected_results(), self.get_predictions())

    def calculate_accuracy(self):
        """Get the accuracy for this sample group."""
        self.accuracy = self.predictions.get_accuracy()

    def set_metric_value(self, name, value):
        self.metric_name = name
        self.metric_value = value

    def to_dict(self):
        """Returns the main attributes of this sample group as a dictionary."""
        dictionary = {"sample_group_id": self.id, "accuracy": self.accuracy}
        return dictionary
