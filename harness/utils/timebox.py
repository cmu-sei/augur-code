
class TimeBox:
    """Represents a bounded subset of data over which metrics are calculated."""
    id = 0
    size = 0
    accuracy = 0
    metric_value = 0
    dataset = None
    starting_idx = 0

    def __init__(self, timebox_id, timebox_size):
        self.id = timebox_id
        self.size = timebox_size

    def set_data(self, dataset, starting_idx):
        self.dataset = dataset
        self.starting_idx = starting_idx

    def get_output(self):
        """Returns the output for this timebox, slicing it from the dataset given the start idx and size."""
        return self.dataset.get_output()[self.starting_idx, self.starting_idx + self.size]

    def calculate_accuracy(self):
        """Calculate the accuracy with the given data and store it."""
        # TODO: implement accuracy calculation.
        accuracy = 0

    def set_metric_value(self, value):
        self.metric_value = value