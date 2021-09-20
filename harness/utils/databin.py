import random


def create_bins(bin_info_list):
    """Creates a list of bins based on a string list of bin info."""
    bins = []
    for bin_info in bin_info_list:
        bins.append(DataBin(bin_info[0], bin_info[1]))
    return bins


def sort_into_bins(ids, values, bins):
    """Sorts the ids into bins, based on the given values matching bin values."""
    for sample_idx, value in enumerate(values):
        for bin_idx, bin in enumerate(bins):
            if value == bin.value:
                bins[bin_idx].add(ids[sample_idx])
                break

    return bins


class DataBin:
    """Class to represent a bin of data of a certain classification."""
    name = ""
    value = None
    ids = []
    id_queue = []

    def __init__(self, new_name, new_value):
        self.name = new_name
        self.value = new_value
        self.ids = []

    def setup_queue(self):
        self.id_queue = self.ids.copy()
        random.shuffle(self.id_queue)

    def get_queue_length(self):
        return len(self.id_queue)

    def pop_from_queue(self):
        if len(self.id_queue) > 0:
            return self.id_queue.pop(0)
        else:
            return None

    def add(self, new_id):
        self.ids.append(new_id)

    def get_random(self):
        return random.choice(self.ids)

    def size(self):
        return len(self.ids)

    def info(self):
        return self.name + ": " + str(self.size()) + " samples "

    def to_string(self):
        return self.name + ": " + str(self.ids)
