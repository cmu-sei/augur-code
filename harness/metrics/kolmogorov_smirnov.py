import numpy as np
from scipy.stats import ks_2samp


def metric_error(timebox, dataset, predictions):
    """Calculates the Kolmogorov-Smirnov error."""
    ks_stat = ks_2samp(timebox.get_expected_results(), timebox.get_predictions()).statistic
    print(f"KS Test: {ks_stat}")
    return ks_stat

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
    kstest = metric_error(timebox)
    print(f"Length of return is {np.array(kstest).size}")
    