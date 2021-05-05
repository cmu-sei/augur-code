import math


def get_bin_index(index, curr_bin_idx, num_total_bins, params):
    """This only makes sense with num_total_bins is 2. Otherwise it will return non-alternating results."""
    if math.floor(math.sin(index)) == 0:
        return curr_bin_idx
    else:
        return (curr_bin_idx + 1) % num_total_bins
