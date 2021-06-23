import numpy as np


def metric_error(timebox, dataset, predictions):
    """Calculates the STEPD error."""
    r0 = np.count_nonzero(np.array_equal(dataset.get_output(), predictions))
    n0 = dataset.get_number_of_samples()
    rr = np.count_nonzero(timebox.get_correctness())
    nr = timebox.get_number_of_samples()

    p_hat = (r0 + rr) / (n0 + nr)
    t = abs((r0 / rr) - (n0 / nr)) - 0.5 * (1 / n0 + 1 / nr) / np.sqrt(p_hat * (1 - p_hat) * (1 / n0 + 1 / nr))
    return t
