import random

import numpy

from utils.logging import print_and_log

# Tracks how many samples have been added from each bin for the current timebox.
selected_samples_by_bin = []
last_timebox_id = -1
prevalence_array = None


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
    if prevalence_array is None:
        prepare_prevalence_array(params)
    if timebox_id >= len(prevalence_array):
        raise Exception(f"There are too few prevalences ({len(prevalence_array)}) configured for the timeboxes (currently ({timebox_id+1})")
    target_prevalence = prevalence_array[timebox_id]

    # Get the prevalence we have so far.
    prevalence_bin_id = params.get("prevalence_bin")
    if prevalence_bin_id > num_total_bins:
        raise Exception(f"Bin selected as prevalence forcer is not a valid bin id ({prevalence_bin_id}")
    timebox_size = params.get("timebox_size")

    # Get the next bin id, ensuring we don't go over the target prevalence.
    print_and_log(f"Timebox id: {timebox_id}")
    next_bin_id = get_random_bin_with_prevalence(num_total_bins, timebox_size, prevalence_bin_id, target_prevalence)

    selected_samples_by_bin[next_bin_id] += 1
    last_timebox_id = timebox_id
    return next_bin_id


def prepare_prevalence_array(params):
    """Prepares the prevalences array, loading it from config, and filling repetitions if needed."""
    global prevalence_array
    if prevalence_array is None:
        prevalence_array = params.get("prevalences")
        prevalence_repeat = params.get("prevalence_repeat")
        if prevalence_repeat:
            num_init_prevs = len(prevalence_array)
            num_timeboxes = int(params.get("max_num_samples") / params.get("timebox_size"))
            random_rep_range = params.get("prevalence_repeat_range")
            print_and_log(f"Generating repetitions for timeboxes (total timeboxes: {num_timeboxes})")
            full_prev_array = prevalence_array.copy()
            for i in range(num_init_prevs, num_timeboxes, num_init_prevs):
                repetition_prev_array = [prevalence + random.randrange(random_rep_range[0], random_rep_range[1]) \
                                         for prevalence in prevalence_array]
                full_prev_array += repetition_prev_array
            prevalence_array = full_prev_array
        print_and_log(f"Prevalences array: {prevalence_array}")


def get_random_bin_with_prevalence(num_total_bins, timebox_size, main_prevalence_bin_id, target_prevalence):
    """Gets the next bin randomly, but ensuring we don't go over the max prevalence."""
    all_prevalences = [round(selected_samples_by_bin[curr_id] / timebox_size, 2) * 100 for curr_id in range(num_total_bins)]
    main_prevalence = all_prevalences[main_prevalence_bin_id]
    other_prevalences_sum = sum([prevalence for bin_id, prevalence in enumerate(all_prevalences) if bin_id != main_prevalence_bin_id])
    print_and_log(f"All prevalences: {all_prevalences}, Main prevalence: {main_prevalence}, target: {target_prevalence}")

    # Get random bins, except if 1) we already have enough of the prevalence one, or 2) we got so many of the others
    # that we need to force getting the prevalence one.
    if main_prevalence >= target_prevalence:
        print_and_log("Selecting from non-main bins")
        next_bin_id = randrange_with_exclusions(num_total_bins, [main_prevalence_bin_id])
    elif other_prevalences_sum >= (100 - target_prevalence):
        print_and_log("Selecting main bin")
        next_bin_id = main_prevalence_bin_id
    else:
        print_and_log("Selecting random bin")
        next_bin_id = randrange_with_exclusions(num_total_bins, [])

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
