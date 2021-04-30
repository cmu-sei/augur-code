import random

from utils import dataset as augur_dataset

DRIFT_NAME = "Random"


def generate_drift(databins, params):
    """Generates a drifted dataset by randomly selecting samples from each bin, to a total configured in the params."""
    print(DRIFT_NAME + "Generating drift with params: ")
    print(params)

    drifted_dataset = augur_dataset.DataSet()
    num_samples = params.get("num_samples")
    for i in range(0, num_samples):
        bin_idx = random.randrange(len(databins))
        random_id = databins[bin_idx].get_random()
        drifted_dataset.add_by_reference(random_id)

    print("Finished generating drift!")
    return drifted_dataset
