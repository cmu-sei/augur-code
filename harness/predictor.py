import pandas as pd
import numpy as np

from utils import model as augur_model
from utils import dataset as augur_dataset
from utils.config import Config

CONFIG_FILENAME = "./predictor_config.json"


# Generates predictions based on model and SAR data.
def predict(model, x_band_data, x_angle_data):
    predictions = model.predict([x_band_data, x_angle_data]).flatten()
    print("Predictions shape:", predictions.shape, flush=True)
    return predictions


# Saves the ids, predictions and metrics into a JSON file.
def save_predictions(x_ids, predictions, metrics, output_filename):
    # Turn everything into a DataFrame before turning into JSON.
    print("Creating DataFrame", flush=True)
    output_df = pd.DataFrame()
    output_df["id"] = x_ids
    output_df["prediction"] = predictions

    # Add in the metrics (assuming a dict with them).
    for metric_name in metrics.keys():
        output_df[metric_name] = metrics[metric_name]

    print("Saving DataFrame to JSON file", flush=True)
    output_df.to_json(output_filename, orient="records", indent=4)
    print("Finished saving JSON file", flush=True)


def save_updated_dataset(dataset, predictions, output_filename):
    dataset.y_results = np.round(predictions)
    dataset.save_data(output_filename)


# Main code.
def main():
    # Load config.
    config = Config()
    config.load(CONFIG_FILENAME)

    # Load dataset to predict on.
    dataset = augur_dataset.DataSet()
    dataset.load_data(config.get("dataset"))

    # Load model.
    model = augur_model.load_model_from_file(config.get("model"))
    model.summary()

    # Predict.
    predictions = predict(model, dataset.x_combined_bands, dataset.x_angle)

    # Save to file.
    mode = config.get("mode")
    if mode == "default":
        save_predictions(dataset.x_ids, predictions, {}, config.get("output"))
    else:
        save_updated_dataset(dataset, predictions, config.get("output"))


if __name__ == '__main__':
    main()
