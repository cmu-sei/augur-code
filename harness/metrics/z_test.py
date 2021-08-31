import numpy as np
from scipy.stats import ttest_ind


def metric_error(timebox, dataset, predictions):
    """Calculates the z test statistic."""
    z_stat = ttest_ind(dataset.get_output(), predictions).statistic
    print(f"Z Test: {z_stat}")
    return z_stat

if __name__ == "__main__":
    class Dataset:

        def __init__(self, data):
            self.data = data

        def get_output(self):
            return self.data

    n = 100
    dataset = Dataset(np.random.random(n))
    predictions = np.random.random(n)
    z_stat = metric_error(None, dataset, predictions)
    print(f"Length of return is {np.array(z_stat).size}")
    