import numpy as np
import math
import pandas


class IntervalGenerator:
    """Abstract class with common functions for interval generators."""
    def __init__(self, starting_interval, interval_unit):
        self.starting_interval = starting_interval
        self.interval_unit = interval_unit

    def calculate_time_interval(self, sample_idx):
        """Calculates the time interval number where a sample falls into, given the time step."""
        raise NotImplementedError()

    def get_last_time_interval(self):
        raise NotImplementedError()

    def get_number_of_intervals(self):
        """Returns the number of intervals for the total of samples."""
        interval_delta = (self.get_last_time_interval() - self.starting_interval + pandas.to_timedelta(1, self.interval_unit))
        print(interval_delta)
        return int(interval_delta / pandas.to_timedelta(1, self.interval_unit))


class TimestampIntervalGenerator(IntervalGenerator):
    """Calculates time intervals based on timestamps."""
    DAYS_TO_SECONDS = 1 / (24 * 60 * 60)

    def __init__(self, starting_interval, interval_unit, time_step_in_days, timestamps):
        super().__init__(starting_interval, interval_unit)
        self.time_step = time_step_in_days * self.DAYS_TO_SECONDS
        self.timestamps = timestamps
        self.min_timestamp = timestamps[0]
        self.max_timestamp = timestamps[timestamps.size-1]

    def calculate_time_interval(self, sample_idx):
        """Calculates the time interval number where a timestamp falls into, given the time step."""
        delta = math.floor((self.timestamps[sample_idx] - self.min_timestamp) / self.time_step)
        return self.starting_interval + pandas.Timedelta(delta, unit=self.interval_unit)

    def get_last_time_interval(self):
        return self.calculate_time_interval(self.max_timestamp)


class NumSamplesIntervalGenerator(IntervalGenerator):
    """Calculates time intervals based on number of samples per interval."""

    def __init__(self, starting_interval, interval_unit, samples_per_time, num_samples):
        super().__init__(starting_interval, interval_unit)
        self.samples_per_time = samples_per_time
        self.num_samples = num_samples

    def calculate_time_interval(self, sample_idx):
        """Calculates the time interval number where a sample falls into, given the time step."""
        delta = math.floor(sample_idx / self.samples_per_time)
        return self.starting_interval + pandas.Timedelta(delta, unit=self.interval_unit)

    def get_last_time_interval(self):
        return self.calculate_time_interval(self.num_samples - 1)


class TimeSeries:
    """Represents a time series with a time interval and an aggregated value."""
    time_intervals = []
    aggregated = np.empty(0)
    num_samples = np.empty(0, dtype=int)
    pdf = []
    pdf_params = []

    def check_valid_idx(self, time_interval_idx):
        """Checks if the interval idx is inside the valid range"""
        if time_interval_idx >= self.get_num_intervals() or time_interval_idx < 0:
            raise Exception(f"Invalid time interval id passed: {time_interval_idx}, length is {self.get_num_intervals()}")

    def get_time_intervals(self):
        """Getter for the list of time intervals."""
        return self.time_intervals

    def get_num_intervals(self):
        """Returns the amount of intervals in the object."""
        return len(self.time_intervals)

    def get_interval_index(self, interval):
        """Returns the index of the given interval."""
        interval_idx = self.time_intervals.index(interval)
        return interval_idx

    def get_aggregated(self, time_interval_idx=None):
        """Returns either the full array of aggregated values, or a specific one by the time interval index."""
        if time_interval_idx is None:
            return self.aggregated
        else:
            self.check_valid_idx(time_interval_idx)
            return self.aggregated[time_interval_idx]

    def get_pdf(self, time_interval_idx):
        """Getter for a specific pdf for the given time interval."""
        self.check_valid_idx(time_interval_idx)
        return self.pdf[time_interval_idx]

    def set_pdf(self, pdf):
        """Setter for the list of pdfs."""
        self.pdf = pdf

    def get_pdf_params(self, time_interval_idx):
        """Getter for a specific pdf_params for the given time interval."""
        self.check_valid_idx(time_interval_idx)
        return self.pdf_params[time_interval_idx]

    def set_pdf_params(self, pdf_params):
        """Setter for the list of pdf parameters."""
        self.pdf_params = pdf_params

    def get_num_samples(self, time_interval_idx):
        """Returns the number of samples aggregated for the given time interval."""
        self.check_valid_idx(time_interval_idx)
        return self.num_samples[time_interval_idx]

    def get_model_input(self):
        """Returns the time intervals and aggregated values as a model input."""
        return [self.get_time_intervals(), self.get_aggregated()]

    def setup_time_intervals(self, starting_interval, unit, num_intervals):
        """Sets up time intervals given the starting one, unit, and how many, as well as pre-allocating other arrays."""
        self.time_intervals = [""] * num_intervals
        for i in range(0, num_intervals):
            self.time_intervals[i] = starting_interval + pandas.Timedelta(i, unit=unit)
            print(self.time_intervals[i])

        self.aggregated = np.zeros(self.get_num_intervals(), dtype=int)
        self.num_samples = np.zeros(self.get_num_intervals(), dtype=int)
        self.pdf = [[]] * self.get_num_intervals()
        self.pdf_params = [{}] * self.get_num_intervals()

    def add_data(self, time_interval, aggregated_value, num_samples=None, pdf=None, pdf_params=None):
        """Adds aggregated data to the given time position."""
        if time_interval in self.time_intervals:
            idx = np.where(self.time_intervals == time_interval)
            self.aggregated[idx] = aggregated_value
            self.num_samples[idx] = num_samples
            self.pdf[idx] = pdf
            self.pdf_params[idx] = pdf_params
        else:
            raise Exception(f"Invalid time index, not found in times array: {time_interval}")

    def aggregate(self, starting_interval, interval_unit, dataset, values, interval_generator):
        """Aggregates a given dataset, and stores it in memory."""
        # Pre-allocate space, and fill up times, given timestamps in dataset.
        num_intervals = interval_generator.get_number_of_intervals()
        print(num_intervals)
        self.setup_time_intervals(starting_interval, interval_unit, num_intervals)

        # Go over all samples, adding their output to the corresponding position in the aggregated array.
        total_num_samples = dataset.get_number_of_samples()
        for sample_idx in range(0, total_num_samples):
            # Calculate the interval for the current sample.
            sample_time_interval = interval_generator.calculate_time_interval(sample_idx)

            # Update the aggregated sum and the number of samples.
            interval_idx = self.get_interval_index(sample_time_interval)
            self.aggregated[interval_idx] += values[sample_idx]
            self.num_samples[interval_idx] += 1

    def aggregate_by_timestamp(self, start_interval_string, interval_unit, dataset, values, time_step):
        """Aggregates a given dataset on the given time step, and stores it in memory."""
        start_interval = pandas.to_datetime(start_interval_string)
        interval_generator = TimestampIntervalGenerator(start_interval, interval_unit, time_step, dataset.get_timestamps())
        return self.aggregate(start_interval, interval_unit, dataset, values, interval_generator)

    def aggregate_by_number_of_samples(self, start_interval_string, interval_unit, dataset, values, samples_per_time):
        """Aggregates a given dataset on the given time step, and stores it in memory."""
        start_interval = pandas.to_datetime(start_interval_string)
        interval_generator = NumSamplesIntervalGenerator(start_interval, interval_unit, samples_per_time, dataset.get_number_of_samples())
        return self.aggregate(start_interval, interval_unit, dataset, values, interval_generator)

    def to_dict(self):
        """Returns the main attributes of this object as a dictionary."""
        dictionary = {"time_intervals": self.time_intervals,
                      "aggregated": self.aggregated,
                      "num_samples": self.num_samples,
                      "pdf": self.pdf,
                      "pdf_params": self.pdf_params}
        return dictionary


def create_test_time_series(dist_start=0, dist_end=10, dist_total=10):
    """A manually created time series for testing."""
    class FakeDataset:
        def __init__(self, num_samples):
            self.num_samples = num_samples
        def get_number_of_samples(self):
            return self.num_samples

    samples_per_time = 4
    output_values = np.array([1, 2, 1, 3, 4, 5, 6, 1, 2, 1, 3, 4, 5, 6, 1])
    dataset = FakeDataset(output_values.size)

    time_series = TimeSeries()
    time_series.aggregate_by_number_of_samples(pandas.to_datetime("2021-11-01"), "days", dataset, output_values, samples_per_time)
    time_series.set_pdf([np.random.randint(dist_start, dist_end, (dist_total))] * time_series.get_num_intervals())
    time_series.set_pdf_params([{"mean": 5, "std_dev": 3}] * time_series.get_num_intervals())

    return time_series


def test():
    """Quick test of a timeseries aggregation."""
    print("Testing")

    expected_values = np.array([7, 16, 10, 12])
    time_series = create_test_time_series()
    print(time_series.to_dict())
    print(f"Expected values: {expected_values}")


if __name__ == '__main__':
    test()
