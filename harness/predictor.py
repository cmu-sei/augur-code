import json
import os
import shutil
import datetime

from datasets.timeseries import TimeSeries
from training import model_utils
from training.predictions import Predictions
from datasets import ref_dataset
from utils.config import Config
from utils import arguments
from utils import logging
from utils.logging import print_and_log
from datasets import dataset
import metrics.base_metric as augur_metrics

DEFAULT_CONFIG_FILENAME = "./predictor_config.json"
METRIC_EXP_CONFIG_FOLDER = "../experiments/predictor"
PACKAGED_FOLDER_BASE = "../output/packaged/"
LOG_FILE_NAME = "predictor.log"


def load_datasets(input_config):
    """Based on the config, loads the dataset to use, and the reference one if needed."""
    dataset_class = dataset.load_dataset_class(input_config.get("dataset_class"))
    reference_dataset = None
    if "base_dataset" in input_config:
        full_dataset, reference_dataset = ref_dataset.load_full_from_ref_and_base(dataset_class, input_config.get("dataset"), input_config.get("base_dataset"))
    else:
        full_dataset = dataset_class()
        full_dataset.load_from_file(input_config.get("dataset"))

    return full_dataset, reference_dataset


def predict(model, model_input, threshold):
    """Generates predictions based on model, returns object with raw and classified predictions."""
    raw_predictions = model.predict(model_input).flatten()
    print_and_log(f"Predictions shape: {raw_predictions.shape}")
    predictions = Predictions(threshold)
    predictions.store_raw_predictions(raw_predictions)
    predictions.classify_raw_predictions()
    return predictions


def ts_predict(model, model_input, threshold):
    """Generates predictions based on model, returns object with raw and classified predictions."""
    raw_predictions = model.predict(model_input).flatten()
    print_and_log(f"Predictions shape: {raw_predictions.shape}")
    ts_predictions = TimeSeries()
    # TODO: load the raw_predictions somehow in the TimeSeries result.
    return ts_predictions


def calculate_metrics(time_series, ts_predictions, config):
    """Calculates metrics for the given configs and dataset."""
    if not config.contains("metrics"):
        print_and_log("No metrics configured.")
        return None

    # TODO: Accuracy is not calculated, maybe move to separate function before this one?
    metrics = config.get("metrics")
    results = {}
    for metric_info in metrics:
        metric_name = metric_info.get('name')
        print_and_log(f"Loading metric: {metric_name}")
        metric = augur_metrics.create_metric(metric_info)
        metric.load_metric_functions(metric_info)
        metric.initial_setup(time_series, ts_predictions)

        for interval_index, time_interval in enumerate(time_series.get_time_intervals()):
            # Calculate metric.
            metric.step_setup(interval_index)
            metric_value = metric.calculate_metric()

            # Accumulate results.
            if interval_index not in results.keys():
                results[interval_index] = {}
                results[interval_index]["metrics"] = []
            results[interval_index]["metrics"].append({"name": metric_name, "value": metric_value})

    return results


def save_predictions(full_dataset, predictions, output_filename, reference_dataset=None):
    """Saves the ids, predictions and metrics into a JSON file."""
    print_and_log("Creating predictions DataFrame")
    if reference_dataset:
        output_df = reference_dataset.as_dataframe()
    else:
        output_df = full_dataset.as_dataframe(include_all_data=False)

    predictions.save_to_file(output_filename, output_df)


def save_metrics(metrics, metrics_filename):
    """Stores the given metrics to an output."""
    if len(metrics) == 0:
        print_and_log("No metrics to store to file.")
        return

    print_and_log("Saving metrics to JSON file")
    with open(metrics_filename, "w") as outfile:
        json.dump(metrics, outfile, indent=4)
    print_and_log("Finished saving JSON file")


def save_updated_dataset(updated_dataset, predictions, output_filename):
    """Saves a dataset to a JSON file, adding the given predictions first."""
    updated_dataset.set_output(predictions)
    updated_dataset.save_to_file(output_filename)


def package_results(config):
    """Copies all results to a date-time folder to store experiment results."""
    dataset = config.get("input").get("dataset")
    model = config.get("input").get("model")
    predictions = config.get("output").get("predictions_output")
    metrics = config.get("output").get("metrics_output")

    # Create time-stamped folder to store results.
    print_and_log("Storing exp results in folder.")
    exp_descriptor = os.path.splitext(os.path.basename(config.config_filename))[0]
    package_folder_name = "exp-" + exp_descriptor + "-" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    full_folder_path = os.path.join(PACKAGED_FOLDER_BASE, package_folder_name)
    if os.path.exists(full_folder_path):
        shutil.rmtree(full_folder_path)
    os.makedirs(full_folder_path)

    # Create subfolder for config.
    config_folder = os.path.join(full_folder_path, "predictor_config")
    os.makedirs(config_folder)

    # Copy all files to the folder to be zipped.
    shutil.copy(config.config_filename, config_folder)
    shutil.copy(dataset, full_folder_path)
    shutil.copytree(model, os.path.join(full_folder_path, os.path.basename(os.path.normpath(model))))
    shutil.copy(predictions, full_folder_path)
    shutil.copy(metrics, full_folder_path)
    shutil.copy(LOG_FILE_NAME, full_folder_path)

    # Create zip as well.
    print_and_log("Storing exp results in zip file.")
    shutil.make_archive(full_folder_path, "zip", full_folder_path)


# Main code.
def main():
    logging.setup_logging(LOG_FILE_NAME)

    # Allow selecting configs for experiments, and load it.
    args = arguments.get_parsed_arguments()
    config_file = Config.get_config_file(args, METRIC_EXP_CONFIG_FOLDER, DEFAULT_CONFIG_FILENAME)
    config = Config()
    config.load(config_file)

    # Load dataset to predict on (and base one if needed).
    full_dataset, reference_dataset = load_datasets(config.get("input"))

    # Load model.
    model = model_utils.load_model_from_file(config.get("input").get("model"))
    model.summary()

    # Load TS model.
    print_and_log("Loading time-series model")
    ts_model = model_utils.load_model_from_file(config.get("input").get("ts_model"))
    print_and_log("Time-series model Loaded.")

    # Predict.
    predictions = predict(model, full_dataset.get_model_input(), config.get("threshold"))

    # Save to file, depending on mode, and calculate metrics if needed.
    mode = config.get("mode")
    if mode == "predict":
        predictions.store_expected_results(full_dataset.get_output())
        save_predictions(full_dataset, predictions, config.get("output").get("predictions_output"), reference_dataset)

        # Aggregate dataset and run time series model.
        time_series = TimeSeries()
        time_series.aggregate_by_timestamp(full_dataset, predictions.get_predictions(),
                                           config.get("hyper_parameters").get("time_step"))
        ts_predictions = ts_predict(ts_model, time_series.get_model_input(), config.get("threshold"))

        # Calculate and store metrics.
        metric_results = calculate_metrics(time_series, ts_predictions, config)
        save_metrics(metric_results, config.get("output").get("metrics_output"))

        # If requested, package this experiment results.
        if args.store:
            package_results(config)
    elif mode == "label":
        save_updated_dataset(full_dataset, predictions.get_predictions(), config.get("output").get("labelled_output"))
    else:
        print_and_log("Unsupported mode: " + mode)


if __name__ == '__main__':
    main()
