import numpy as np
from scipy.stats import ttest_ind


def metric_error(sample_group, dataset=None, predictions=None):
    """Calculates the z test statistic."""
    z_stat = ttest_ind(sample_group.get_expected_results(), sample_group.get_predictions()).statistic
    print(f"Z Test: {z_stat}")
    return z_stat


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
    z_stat = metric_error(sample_group)
    print(f"Length of return is {np.array(z_stat).size}")
    