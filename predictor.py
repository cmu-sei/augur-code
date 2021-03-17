import json
import codecs

from utils import model as augur_model
from utils import dataset as augur_dataset


# Generates predictions based on model and SAR data.
def predict(model, x_band_data, x_angle_data):
    predictions = model.predict([x_band_data, x_angle_data])
    print("predictions shape:", predictions.shape)
    return predictions


def save_output(x_ids, predictions, metrics):
    # Link ids to predictions.
    ids = x_ids.tolist()
    predictions_list = predictions.tolist()
    index = 0
    results = []
    for prediction_item in predictions_list:
        result_dict = {"prediction": prediction for prediction in prediction_item}
        result_dict["id"] = ids[index]
        results.append(result_dict)
        index += 1

    # Save results.
    json.dump(results, codecs.open("./output/predictions.json", 'w', encoding='utf-8'), separators=(',', ': '),
              sort_keys=True, indent=4)

# Main code.
def main():
    # Load dataset to predict on.
    [x_ids, x_band_data, x_angle_data, y_all] = augur_dataset.load_data("./input/train.json")

    # Load model.
    model = augur_model.load_model_from_file("./output/trained_model")
    model.summary()

    # Predict.
    predictions = predict(model, x_band_data, x_angle_data)

    # Save to file.
    save_output(x_ids, predictions, [])


if __name__ == '__main__':
    main()
