import numpy as np
from scipy.stats import ks_2samp


def metric_error(time_interval_id, time_series, ts_predictions):
    """Calculates the Kolmogorov-Smirnov error."""
    # TODO: Fix this error metric.
    #ks_stat = ks_2samp(sample_group.get_expected_results(), sample_group.get_predictions()).statistic
    ks_stat = 0
    print(f"KS Test: {ks_stat}")
    return ks_stat


if __name__ == "__main__":
    class TestSampleGroup:

        def __init__(self, samples):
            self.samples = samples

        def get_expected_results(self):
            return np.random.random(self.samples)

        def get_predictions(self):
            return np.random.random(self.samples)

    n = 100
    sample_group = TestSampleGroup(n)
    kstest = metric_error(sample_group)
    print(f"Length of return is {np.array(kstest).size}")
    