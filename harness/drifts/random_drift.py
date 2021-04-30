import random


def generate_drift(databins, drifted_dataset, params):
    """Generates a drifted dataset by randomly selecting samples from each bin, to a total configured in the params."""

    num_samples = params.get("num_samples")
    for i in range(0, num_samples):
        bin_idx = random.randrange(len(databins))
        random_id = databins[bin_idx].get_random()
        drifted_dataset.add_by_reference(random_id)

    return drifted_dataset
