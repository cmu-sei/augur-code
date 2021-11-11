import numpy as np
import math


class TimestampIntervalGenerator:
    """Calculates time intervals based on timestamps."""
    DAYS_TO_SECONDS = 1 / (24 * 60 * 60)

    def __init__(self, time_step_in_days, timestamps):
        self.time_step = time_step_in_days * self.DAYS_TO_SECONDS
        self.timestamps = timestamps
        self.min_timestamp = timestamps[0]
        self.max_timestamp = timestamps[timestamps.size-1]

    def calculate_time_interval(self, sample_idx):
        """Calculates the time interval number where a timestamp falls into, given the time step."""
        return math.floor((self.timestamps[sample_idx] - self.min_timestamp) / self.time_step)

    def get_number_of_intervals(self):
        """Returns the number of intervals for the given timestamps."""
        return self.calculate_time_interval(self.max_timestamp) + 1


class NumSamplesIntervalGenerator:
    """Calculates time intervals based on number of samples per interval."""

    def __init__(self, samples_per_time, num_samples):
        self.samples_per_time = samples_per_time
        self.num_samples = num_samples

    def calculate_time_interval(self, sample_idx):
        """Calculates the time interval number where a sample falls into, given the time step."""
        return math.floor(sample_idx / self.samples_per_time)

    def get_number_of_intervals(self):
        """Returns the number of intervals for the total of samples."""
        return self.calculate_time_interval(self.num_samples - 1) + 1


class TimeSeries:
    """Represents a time series with a time interval and an aggregated value."""
    time_intervals = np.empty(0, dtype=int)
    aggregated = np.empty(0)
    pdf = np.empty(0)
    pdf_params = np.empty(0)

    def get_time_intervals(self):
        return self.time_intervals

    def get_aggregated(self, time_interval_id=None):
        """Returns either the full array of aggregated values, or a specific one by the time interval index."""
        if time_interval_id is None:
            return self.aggregated
        else:
            if time_interval_id > len(self.aggregated):
                return self.aggregated[time_interval_id]
            else:
                raise Exception(f"Invalid time interval id passed: {time_interval_id}, length is {len(self.aggregated)}")

    def get_pdf(self, time_interval_id):
        if time_interval_id > len(self.pdf):
            return self.pdf[time_interval_id]
        else:
            raise Exception(f"Invalid time interval id passed: {time_interval_id}, length is {len(self.pdf)}")

    def set_pdf(self, pdf):
        self.pdf = pdf

    def get_pdf_params(self, time_interval_id):
        if time_interval_id > len(self.pdf_params):
            return self.pdf_params[time_interval_id]
        else:
            raise Exception(f"Invalid time interval id passed: {time_interval_id}, length is {len(self.pdf_params)}")

    def get_model_input(self):
        """Returns the time intervals and aggregated values as a model input."""
        return [self.get_time_intervals(), self.get_aggregated()]

    def allocate(self, start_time_interval, num_intervals):
        """Allocates needed space for arrays."""
        self.time_intervals = np.arange(start_time_interval, num_intervals)
        self.aggregated = np.zeros(self.time_intervals.size)
        self.pdf = np.zeros(self.time_intervals.size)
        self.pdf_params = np.zeros(self.time_intervals.size)    #TODO: check this

    def add_data(self, time, aggregated_value, pdf=None, pdf_params=None):
        """Adds aggregated data to the given time position."""
        if time in self.time_intervals:
            idx = np.where(self.time_intervals == time)
            self.aggregated[idx] = aggregated_value
            self.pdf[idx] = pdf
            self.pdf_params[idx] = pdf_params
        else:
            raise Exception(f"Invalid time index, not found in times array: {time}")

    def aggregate(self, dataset, values, interval_generator, start_time_interval=0):
        """Aggregates a given dataset, and stores it in memory."""
        # Pre-allocate space, and fill up times, given timestamps in dataset.
        num_samples = dataset.get_number_of_samples()
        num_intervals = interval_generator.get_number_of_intervals()
        max_time_interval = start_time_interval + num_intervals - 1
        self.allocate(start_time_interval, num_intervals)

        # Go over all samples, adding their output to the corresponding position in the aggregated array.
        for sample_idx in range(0, num_samples):
            sample_time = interval_generator.calculate_time_interval(sample_idx)
            if start_time_interval <= sample_time <= max_time_interval:
                self.aggregated[sample_time] += values[sample_idx]

    def aggregate_by_timestamp(self, dataset, values, time_step_in_days, start_time=0):
        """Aggregates a given dataset on the given time step, and stores it in memory."""
        interval_generator = TimestampIntervalGenerator(time_step_in_days, dataset.get_timestamps())
        return self.aggregate(dataset, values, interval_generator, start_time)

    def aggregate_by_number_of_samples(self, dataset, values, samples_per_time, start_time=0):
        """Aggregates a given dataset on the given time step, and stores it in memory."""
        interval_generator = NumSamplesIntervalGenerator(samples_per_time, dataset.get_number_of_samples())
        return self.aggregate(dataset, values, interval_generator, start_time)


def test():
    """Quick test of a timeseries aggregation."""
    print("Testing")
    samples_per_time = 4
    output_values = np.array([1, 2, 1, 3, 4, 5, 6, 1, 2, 1, 3, 4, 5, 6, 1])
    expected_values = np.array([7, 16, 10, 12])

    class FakeDataset:
        def __init__(self, num_samples):
            self.num_samples = num_samples
        def get_number_of_samples(self):
            return self.num_samples

    dataset = FakeDataset(output_values.size)
    time_series = TimeSeries()
    time_series.aggregate_by_number_of_samples(dataset, output_values, samples_per_time)
    print(time_series.time_intervals)
    print(time_series.aggregated)
    print(expected_values)


if __name__ == '__main__':
    test()
