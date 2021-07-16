import sys
import json

from training import model_utils
from training.predictions import Predictions
from datasets import ref_dataset
from utils.config import Config
from utils import logging
from utils.logging import print_and_log
from datasets import dataset
from utils.timebox import TimeBox
import metrics.base_metric as augur_metrics

DEFAULT_CONFIG_FILENAME = "./predictor_config.json"
METRIC_EXP_CONFIG_FOLDER = "../experiments/metric"


def load_datasets(config):
    """Based on the config, loads the dataset to use, and the reference one if needed."""
    dataset_class = dataset.load_dataset_class(config.get("dataset_class"))
    reference_dataset = None
    if config.contains("base_dataset"):
        full_dataset = ref_dataset.load_full_from_ref_and_base(dataset_class, config.get("dataset"), config.get("base_dataset"))
    else:
        full_dataset = dataset_class()
        full_dataset.load_from_file(config.get("dataset"))

    return full_dataset, reference_dataset


def predict(model, model_input):
    """Generates predictions based on model and SAR data."""
    predictions = model.predict(model_input).flatten()
    print_and_log(f"Predictions shape: {predictions.shape}")
    return predictions


def calculate_metrics(dataset, predictions, config):
    """Calculates metrics for the given configs and dataset."""
    timebox_size = int(config.get("timebox_size"))
    metrics = config.get("metrics")
    results = {}
    timeboxes = {}
    for metric_info in metrics:
        metric_name = metric_info.get('name')
        print(f"Loading metric: {metric_name}")
        metric = augur_metrics.create_metric(metric_info)
        metric.load_metric_functions(metric_info)
        metric.initial_setup(dataset, predictions.get_predictions())

        curr_sample_idx = 0
        timebox_id = 0
        while curr_sample_idx < dataset.get_number_of_samples():
            # Set up timebox.
            if timebox_id not in timeboxes.keys():
                timebox = TimeBox(timebox_id, timebox_size)
                timebox.set_data(predictions, curr_sample_idx)
                timebox.calculate_accuracy()
                timeboxes[timebox_id] = timebox

            # Calculate metric.
            curr_timebox = timeboxes[timebox_id]
            metric.step_setup(curr_timebox)
            metric_value = metric.calculate_metric()
            curr_timebox.set_metric_value(metric_name, metric_value)

            # Accumulate results.
            if curr_timebox.id not in results.keys():
                results[curr_timebox.id] = curr_timebox.to_dict()
                results[curr_timebox.id]["metrics"] = []
            results[curr_timebox.id]["metrics"].append({"name": metric_name, "value": curr_timebox.metric_value})

            # Update for next cycle.
            curr_sample_idx += timebox_size
            timebox_id += 1

    return results


def save_predictions(full_dataset, predictions, output_filename, reference_dataset=None):
    """Saves the ids, predictions and metrics into a JSON file."""
    # Turn everything into a DataFrame before turning into JSON.
    print_and_log("Creating predictions DataFrame")
    if reference_dataset:
        output_df = reference_dataset.as_dataframe()
    else:
        output_df = full_dataset.as_basic_dataframe()
    output_df["truth"] = full_dataset.y_output
    output_df["prediction"] = predictions.get_predictions()
    output_df["raw_prediction"] = predictions.get_raw_predictions()

    print_and_log("Saving predictions DataFrame to JSON file")
    output_df.to_json(output_filename, orient="records", indent=4)
    print_and_log("Finished saving predictions JSON file")


def save_metrics(metrics, metrics_filename):
    """Stores the given metrics to an output."""
    if len(metrics) == 0:
        print("No metrics to store to file.")
        return

    print_and_log("Saving metrics to JSON file")
    with open(metrics_filename, "w") as outfile:
        json.dump(metrics, outfile, indent=4)
    print_and_log("Finished saving JSON file")


def save_updated_dataset(updated_dataset, predictions, output_filename):
    """Saves a dataset to a JSON file, adding the given predictions first."""
    updated_dataset.set_output(predictions)
    updated_dataset.save_to_file(output_filename)


# Main code.
def main():
    logging.setup_logging("predictor.log")

    # Allow selecting configs for experiments, and load it.
    config_file = Config.choose_from_folder(sys.argv, METRIC_EXP_CONFIG_FOLDER, DEFAULT_CONFIG_FILENAME)
    config = Config()
    config.load(config_file)

    # Load dataset to predict on (and base one if needed).
    full_dataset, reference_dataset = load_datasets(config)

    # Load model.
    model = model_utils.load_model_from_file(config.get("model"))
    model.summary()

    # Predict.
    raw_predictions = predict(model, full_dataset.get_model_input())
    predictions = Predictions(config.get("threshold"))
    predictions.store_raw_predictions(raw_predictions)
    predictions.store_expected_results(full_dataset.get_output())

    # Save to file, depending on mode, and calculate metrics if needed.
    mode = config.get("mode")
    if mode == "predict":
        metric_results = calculate_metrics(full_dataset, predictions, config)
        save_predictions(full_dataset, predictions, config.get("output"), reference_dataset)
        save_metrics(metric_results, config.get("metrics_output"))
    elif mode == "label":
        save_updated_dataset(full_dataset, predictions.get_predictions(), config.get("output"))
    else:
        print_and_log("Unsupported mode: " + mode)


if __name__ == '__main__':
    main()
