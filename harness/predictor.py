import sys

import pandas as pd
import numpy as np

from utils import model as augur_model
from utils import dataset as augur_dataset
from utils.config import Config

DEFAULT_CONFIG_FILENAME = "./predictor_config.json"
METRIC_EXP_CONFIG_FOLDER = "../experiments/metric"


# Generates predictions based on model and SAR data.
def predict(model, x_band_data, x_angle_data):
    predictions = model.predict([x_band_data, x_angle_data]).flatten()
    print("Predictions shape:", predictions.shape, flush=True)
    return predictions


# Generates predictions based on model and SAR data.
def classify(predictions, threshold):
    return np.where(predictions > threshold, 1, 0)


# Saves the ids, predictions and metrics into a JSON file.
def save_predictions(dataset, predictions, output_filename):
    # Turn everything into a DataFrame before turning into JSON.
    print("Creating predictions DataFrame", flush=True)
    output_df = pd.DataFrame()
    output_df["id"] = dataset.x_ids
    output_df["original_id"] = dataset.x_original_ids
    output_df["truth"] = dataset.y_output
    output_df["prediction"] = predictions

    print("Saving predictions DataFrame to JSON file", flush=True)
    output_df.to_json(output_filename, orient="records", indent=4)
    print("Finished saving predictions JSON file", flush=True)


def save_metrics(metrics, metrics_filename):
    print("Creating DataFrame", flush=True)
    output_df = pd.DataFrame()

    # Add in the metrics (assuming a dict with them).
    for metric_name in metrics.keys():
        output_df[metric_name] = metrics[metric_name]

    print("Saving DataFrame to JSON file", flush=True)
    output_df.to_json(metrics_filename, orient="records", indent=4)
    print("Finished saving JSON file", flush=True)


# Saves a dataset to a JSON file, adding the given predictions first.
def save_updated_dataset(dataset, predictions, output_filename):
    dataset.y_output = predictions
    dataset.save_data(output_filename)


# Main code.
def main():
    # Allow selecting configs for experiments, and load it.
    config_file = Config.choose_from_folder(sys.argv, METRIC_EXP_CONFIG_FOLDER, DEFAULT_CONFIG_FILENAME)
    config = Config()
    config.load(config_file)

    # Load dataset to predict on (and base one if needed).
    dataset = augur_dataset.DataSet()
    base_dataset = None
    if config.contains("base_dataset"):
        base_dataset = config.get("base_dataset")
        print("Base dataset: " + base_dataset)
    dataset.load_data(config.get("dataset"), base_dataset)

    # Load model and metrics.
    model = augur_model.load_model_from_file(config.get("model"))
    #if config.contains("metrics"):
    #augur_model.add_metrics(model, config.get("metrics"))
    model.summary()

    # Predict.
    predictions = predict(model, dataset.x_combined_bands, dataset.x_angle)
    classified = classify(predictions, config.get("threshold"))

    # Save to file.
    mode = config.get("mode")
    if mode == "predict":
        save_predictions(dataset, classified, config.get("output"))
    elif mode == "label":
        save_updated_dataset(dataset, classified, config.get("output"))
    else:
        print("Unsupported mode: " + mode)


if __name__ == '__main__':
    main()
