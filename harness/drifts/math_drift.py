import math


def get_bin_index(index, timebox_id, curr_bin_idx, num_total_bins, params):
    """This only makes sense with num_total_bins is 2. Otherwise it will return alternating between 2 continguous indexes, increasing per timebox"""
    if math.floor(math.sin(index)) == 0:
        return curr_bin_idx
    else:
        return (curr_bin_idx + 1) % num_total_bins


def test(full_dataset, params):
    raise NotImplementedError("Test not implemented for this drift.")
