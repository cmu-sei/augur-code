import random

# Tracks how many samples have been added from each bin for the current timebox.
selected_samples_by_bin = []


def get_bin_index(sample_index, timebox_id, curr_bin_idx, num_total_bins, params):
    """Selects samples to have a given prevalence between options."""
    # Initialize the sample tracker.
    global selected_samples_by_bin
    if len(selected_samples_by_bin) == 0:
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
    curr_prevalence = selected_samples_by_bin[prevalence_bin_id] / timebox_size

    # Get the next bin id, ensuring we don't go over the target prevalence.
    if curr_prevalence < target_prevalence:
        next_bin_id = prevalence_bin_id
    else:
        exclusions = [prevalence_bin_id]
        next_bin_id = randrange_with_exclusions(num_total_bins, exclusions)
    return next_bin_id


def randrange_with_exclusions(range_max, exclusions):
    """Recursive function to get random values from a range avoiding excluded items."""
    selection = random.randrange(range_max)
    return randrange_with_exclusions(range_max, exclusions) if selection in exclusions else selection