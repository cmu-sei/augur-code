import numpy as np
import math


class TimeSeries:
    """Represents a time series with a time and an aggregated value."""
    times = np.empty(0)
    aggregated = np.empty(0)

    def get_times(self):
        return self.times

    def get_aggregated(self):
        return self.aggregated

    def allocate(self, start_time, max_time):
        """Allocates needed space for arrays."""
        self.times = np.arange(start_time, max_time)
        self.aggregated = np.zeros(self.times.size)

    def add_data(self, time, aggregated_value):
        """Adds aggregated data to the given time position."""
        if time in self.times:
            idx = np.where(self.times == time)
            self.aggregated[idx] = aggregated_value
        else:
            raise Exception(f"Invalid time index, not found in times array: {time}")

    def calculate_time(self, timestamp, init_timestamp, time_step):
        """Calculates the time interval number where a timestamp falls into, given the time step."""
        return math.floor((timestamp - init_timestamp) / time_step)

    def aggregate(self, dataset, time_step_in_days, start_time=0):
        """Aggregates a given dataset on the given time step, and stores it in memory."""
        DAYS_TO_SECONDS = 1 / (24 * 60 * 60)
        time_step = time_step_in_days * DAYS_TO_SECONDS

        # Pre-allocate space, and fill up times, given timestamps in dataset.
        num_samples = dataset.get_number_of_samples()
        timestamps = dataset.get_timestamps()
        min_timestamp = timestamps[0]
        max_timestamp = timestamps[num_samples-1]
        max_time = start_time + self.calculate_time(max_timestamp, min_timestamp, time_step)
        self.allocate(start_time, max_time)

        # Go over all samples, adding their output to the corresponding position in the aggregated array.
        outputs = dataset.get_output()
        for sample_idx in range(0, num_samples):
            sample_time = self.calculate_time(timestamps[sample_idx], min_timestamp, time_step)
            if start_time <= sample_time < max_time:
                self.aggregated[sample_time] += outputs[sample_idx]
