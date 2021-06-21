
class TimeBox:
    """Represents a bounded subset of data over which metrics are calculated."""
    id = 0
    size = 0
    accuracy = 0
    metric_value = 0
    data = None

    def __init__(self, timebox_id, timebox_size):
        self.id = timebox_id
        self.size = timebox_size

    def set_data(self, dataset, starting_idx):
        # TODO: implement this split.
        self.data = dataset.split(starting_idx, self.size)

    def calculate_accuracy(self):
        """Calculate the accuracy with the given data and store it."""
        # TODO: implement accuracy calculation.
        accuracy = 0

    def set_metric_value(self, value):
        self.metric_value = value
