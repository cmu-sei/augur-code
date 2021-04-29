import random


def sort_into_bins(ids, outputs, bin_names, bin_values):
    """Sorts the ids into bins, based on their outputs matching bin values."""
    bins = []
    for bin_name in bin_names:
        bins.append(DataBin(bin_name))

    # Go over all samples and then check which bin value matches it, to move sample into that bin.
    for sample_idx, output in enumerate(outputs):
        for bin_idx, bin_value in enumerate(bin_values):
            if output == bin_value:
                bins[bin_idx].add(ids[sample_idx])
                break

    return bins


class DataBin:
    """Class to represent a bin of data of a certain classification."""
    type = ""
    ids = []

    def __init__(self, new_type):
        self.type = new_type
        self.ids = []

    def add(self, new_id):
        self.ids.append(new_id)

    def get_random(self):
        return random.choice(self.ids)

    def size(self):
        return len(self.ids)

    def info(self):
        return self.type + ": " + str(self.size()) + " samples "

    def to_string(self):
        return self.type + ": " + str(self.ids)
