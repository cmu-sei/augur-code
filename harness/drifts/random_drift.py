import random


def get_bin_index(index, curr_bin_idx, num_total_bins, params):
    """Generates a drifted dataset by randomly selecting samples from each bin, to a total configured in the params."""
    bin_idx = random.randrange(num_total_bins)
    return bin_idx