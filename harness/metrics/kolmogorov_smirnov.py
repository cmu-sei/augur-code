import numpy as np
from scipy.stats import ks_2samp


def metric_error(timebox, dataset, predictions):
    """Calculates the Kolmogorov-Smirnov error."""
    ks_stat = ks_2samp(dataset.get_output(), predictions).statistic
    print(f"KS Test: {ks_stat}")
    return ks_stat

if __name__ == "__main__":
    class Dataset:

        def __init__(self, data):
            self.data = data

        def get_output(self):
            return self.data

    n = 100
    dataset = Dataset(np.random.random(n))
    predictions = np.random.random(n)
    kstest = metric_error(None, dataset, predictions)
    print(f"Length of return is {np.array(kstest).size}")
    