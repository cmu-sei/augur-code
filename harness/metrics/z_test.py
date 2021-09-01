import numpy as np
from scipy.stats import ttest_ind


def metric_error(timebox, dataset, predictions):
    """Calculates the z test statistic."""
    z_stat = ttest_ind(timebox.get_expected_results(), timebox.get_predictions()).statistic
    print(f"Z Test: {z_stat}")
    return z_stat

if __name__ == "__main__":
    class Timebox:

        def __init__(self, samples):
            self.samples = samples

        def get_expected_results(self):
            return np.random.random(self.samples)

        def get_predictions(self):
            return np.random.random(self.samples)

    n = 100
    timebox = Timebox(n)
    z_stat = metric_error(timebox)
    print(f"Length of return is {np.array(z_stat).size}")
    