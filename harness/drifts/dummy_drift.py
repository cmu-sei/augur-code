
from utils import dataset as augur_dataset

DRIFT_NAME = "Dummy"


def generate_drift(dataset, params):
    print(DRIFT_NAME + "Generating drift with params: ")
    print(params)
    drifted_dataset = augur_dataset.DataSet()
    print("Finished generating dummy drift!")
    return drifted_dataset
