import json
import codecs

from utils import model as augur_model
from utils import dataset as augur_dataset


# Generates predictions based on model and SAR data.
def predict(model, x_band_data, x_angle_data):
    predictions = model.predict([x_band_data, x_angle_data])
    print("predictions shape:", predictions.shape)
    return predictions


# Main code.
def main():
    # Load dataset to predict on.
    [x_band_data, x_angle_data, y_all] = augur_dataset.load_data("./input/train.json")

    # Load model.
    model = augur_model.load_model_from_file("./output/trained_model")
    model.summary()

    # Predict.
    predictions = predict(model, x_band_data, x_angle_data)

    # Save predictions.
    predictions_list = predictions.tolist()
    json.dump(predictions_list, codecs.open("./output/predictions.json", 'w', encoding='utf-8'), separators=(',', ':'),
                                            sort_keys=True, indent=4)


if __name__ == '__main__':
    main()
