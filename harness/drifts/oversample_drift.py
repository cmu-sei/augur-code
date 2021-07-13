import random

import numpy

# Tracks how many samples have been added from each bin for the current timebox.
selected_samples_by_bin = []
last_timebox_id = -1


def get_bin_index(sample_index, timebox_id, curr_bin_idx, num_total_bins, params):
    """Selects samples to have a given prevalence between options."""
    # Initialize the sample tracker each time we change timebox.
    global selected_samples_by_bin
    global last_timebox_id
    if last_timebox_id != timebox_id:
        selected_samples_by_bin = []
        for i in range(0, num_total_bins):
            selected_samples_by_bin.append(0)

    # Get the current target prevalence we are using.
    prevalence_array = params.get("prevalences")
    if timebox_id >= len(prevalence_array):
        raise Exception(f"There are too few prevalences ({len(prevalence_array)}) configured for the timeboxes (currently ({timebox_id+1})")
    target_prevalence = prevalence_array[timebox_id]

    # Get the prevalence we have so far.
    prevalence_bin_id = params.get("prevalence_bin")
    if prevalence_bin_id > num_total_bins:
        raise Exception(f"Bin selected as prevalence forcer is not a valid bin id ({prevalence_bin_id}")
    timebox_size = params.get("timebox_size")
    curr_prevalence = selected_samples_by_bin[prevalence_bin_id] / timebox_size * 100

    # Get the next bin id, ensuring we don't go over the target prevalence.
    print(f"TB: {timebox_id}, Curr prevalence: {curr_prevalence}, target: {target_prevalence}")
    if curr_prevalence < target_prevalence:
        next_bin_id = prevalence_bin_id
    else:
        exclusions = [prevalence_bin_id]
        next_bin_id = randrange_with_exclusions(num_total_bins, exclusions)

    selected_samples_by_bin[next_bin_id] += 1
    last_timebox_id = timebox_id
    return next_bin_id


def randrange_with_exclusions(range_max, exclusions):
    """Recursive function to get random values from a range avoiding excluded items."""
    selection = random.randrange(range_max)
    return randrange_with_exclusions(range_max, exclusions) if selection in exclusions else selection


def test(full_dataset, params):
    """Tests that the given dataset was properly drifted."""
    prevalence_array = params.get("prevalences")
    prevalence_bin_id = params.get("prevalence_bin")
    timebox_size = params.get("timebox_size")
    num_samples = full_dataset.get_number_of_samples()
    num_timeboxes = int(num_samples / timebox_size)

    real_prevalences = [0] * num_timeboxes
    labelled_output = full_dataset.get_output()
    for curr_timebox_id in range(0, num_timeboxes):
        curr_timebox_starting_idx = curr_timebox_id * timebox_size
        timebox_samples = labelled_output[curr_timebox_starting_idx:curr_timebox_starting_idx + timebox_size]
        unique, counts = numpy.unique(timebox_samples, return_counts=True)
        prevalences = (counts / timebox_size * 100).astype(int)
        timebox_prevalences = dict(zip(unique, prevalences))

        print(f"Timebox {curr_timebox_id} real prevalences: {timebox_prevalences}")
        real_prevalences[curr_timebox_id] = timebox_prevalences.get(prevalence_bin_id)

    print(f"Expected prevalences by timebox for bin {prevalence_bin_id}: {prevalence_array}")
    print(f"Obtained prevalences by timebox for bin {prevalence_bin_id}: {real_prevalences}")
