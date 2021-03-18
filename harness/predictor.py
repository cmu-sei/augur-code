import json
import codecs

from utils import model as augur_model
from utils import dataset as augur_dataset
from utils.config import Config

CONFIG_FILENAME = "./predictor_config.json"


# Generates predictions based on model and SAR data.
def predict(model, x_band_data, x_angle_data):
    predictions = model.predict([x_band_data, x_angle_data])
    print("predictions shape:", predictions.shape)
    return predictions


def save_predictions(x_ids, predictions, metrics, output_filename):
    # Link ids to predictions.
    ids = x_ids.tolist()
    predictions_list = predictions.tolist()
    index = 0
    results = []
    for prediction_item in predictions_list:
        result_dict = {"prediction": prediction for prediction in prediction_item}
        result_dict["id"] = ids[index]

        # Add in the metrics (assuming a dict with them).
        for metric_name in metrics:
            result_dict[metric_name] = metrics[metric_name]

        results.append(result_dict)
        index += 1

    # Save results.
    json.dump(results, codecs.open(output_filename, 'w', encoding='utf-8'), separators=(',', ': '),
              sort_keys=True, indent=4)


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
    #else:
    #    save_updated_dataset(dataset)


if __name__ == '__main__':
    main()
