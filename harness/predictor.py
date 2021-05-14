import sys

import pandas as pd
import numpy as np

from utils import model_utils
from datasets import ref_dataset
from utils.config import Config
from utils import logging
from utils.logging import print_and_log

from datasets.iceberg import iceberg_dataset

DEFAULT_CONFIG_FILENAME = "./predictor_config.json"
METRIC_EXP_CONFIG_FOLDER = "../experiments/metric"


def predict(model, model_input):
    """Generates predictions based on model and SAR data."""
    print(model_input)
    predictions = model.predict(model_input).flatten()
    print_and_log(f"Predictions shape: {predictions.shape}")
    return predictions


def classify(predictions, threshold):
    """Generates predictions based on model and SAR data."""
    return np.where(predictions > threshold, 1, 0)


def save_predictions(full_dataset, predictions, output_filename, reference_dataset=None):
    """Saves the ids, predictions and metrics into a JSON file."""
    # Turn everything into a DataFrame before turning into JSON.
    print_and_log("Creating predictions DataFrame")
    output_df = pd.DataFrame()
    if reference_dataset:
        output_df["id"] = reference_dataset.x_ids
        output_df["original_id"] = reference_dataset.x_original_ids
    else:
        output_df["id"] = full_dataset.x_ids
    output_df["truth"] = full_dataset.y_output
    output_df["prediction"] = predictions

    print_and_log("Saving predictions DataFrame to JSON file")
    output_df.to_json(output_filename, orient="records", indent=4)
    print_and_log("Finished saving predictions JSON file")


def save_metrics(metrics, metrics_filename):
    """Stores the given metrics to an output."""
    print_and_log("Creating DataFrame")
    output_df = pd.DataFrame()

    # Add in the metrics (assuming a dict with them).
    for metric_name in metrics.keys():
        output_df[metric_name] = metrics[metric_name]

    print_and_log("Saving DataFrame to JSON file")
    output_df.to_json(metrics_filename, orient="records", indent=4)
    print_and_log("Finished saving JSON file")


def save_updated_dataset(dataset, predictions, output_filename):
    """Saves a dataset to a JSON file, adding the given predictions first."""
    dataset.set_output(predictions)
    dataset.save_to_file(output_filename)


# Main code.
def main():
    logging.setup_logging("predictor.log")

    # Allow selecting configs for experiments, and load it.
    config_file = Config.choose_from_folder(sys.argv, METRIC_EXP_CONFIG_FOLDER, DEFAULT_CONFIG_FILENAME)
    config = Config()
    config.load(config_file)

    # Load dataset to predict on (and base one if needed).
    full_dataset = iceberg_dataset.IcebergDataSet()
    reference_dataset = None
    if config.contains("base_dataset"):
        reference_dataset = ref_dataset.RefDataSet()
        reference_dataset.load_from_file(config.get("dataset"))

        base_dataset = iceberg_dataset.IcebergDataSet()
        base_dataset.load_from_file(config.get("base_dataset"))

        full_dataset = iceberg_dataset.IcebergDataSet()
        full_dataset = reference_dataset.create_from_reference(base_dataset, full_dataset)
    else:
        full_dataset.load_from_file(config.get("dataset"))

    # Load model and metrics.
    model = model_utils.load_model_from_file(config.get("model"))
    # if config.contains("metrics"):
    # augur_model.add_metrics(model, config.get("metrics"))
    model.summary()

    # Predict.
    predictions = predict(model, full_dataset.get_model_input())
    classified = classify(predictions, config.get("threshold"))

    # Save to file.
    mode = config.get("mode")
    if mode == "predict":
        save_predictions(full_dataset, classified, config.get("output"), reference_dataset)
    elif mode == "label":
        save_updated_dataset(full_dataset, classified, config.get("output"))
    else:
        print_and_log("Unsupported mode: " + mode)


if __name__ == '__main__':
    main()
