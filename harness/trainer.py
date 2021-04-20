import sys
import logging

import numpy as np
import sklearn.model_selection as skm
from sklearn.model_selection import KFold

import tensorflow.keras.callbacks as tfcb

from utils import model as augur_model
from utils import dataset as augur_dataset
from utils.config import Config
#from utils import plotter

CONFIG_FILENAME = "./trainer_config.json"
CONFIG = Config()


class TrainingSet(object):
    x_train = []
    y_train = []
    x_validation = None
    y_validation = None

    def has_validation(self):
        return self.x_validation is not None and self.y_validation is not None


def print_and_log(message):
    print(message, flush=True)
    logging.info(message)


def get_callbacks(patience=2):
    file_path = ".model_weights.hdf5"
    es = tfcb.EarlyStopping('val_loss', patience=patience, mode="min")
    msave = tfcb.ModelCheckpoint(file_path, save_best_only=True)
    return [es, msave]


# Split training set into train and validation (75% to actually train)
def split_data(dataset):
    validation_size = CONFIG.get("hyper_parameters").get("validation_size")
    x_band_t, x_band_v, x_angle_t, x_angle_v, y_train, y_validation = skm.train_test_split(dataset.x_combined_bands,
                                                                                           dataset.x_angle,
                                                                                           dataset.y_output,
                                                                                           random_state=42,
                                                                                           test_size=validation_size)
    print("Done splitting validation data from train data", flush=True)
    training_set = TrainingSet()
    training_set.x_train = [x_band_t, x_angle_t]
    training_set.x_validation = [x_band_v, x_angle_v]
    training_set.y_train = y_train
    training_set.y_validation = y_validation
    return training_set


def train(training_set):
    print_and_log("TRAINING")

    model = augur_model.create_model()
    #model.summary()

    epochs = CONFIG.get("hyper_parameters").get("epochs")
    batch_size = CONFIG.get("hyper_parameters").get("batch_size")
    print_and_log(f'Starting training with hyper parameters: epochs: {epochs}, batch size: {batch_size}')

    validation_data = None
    callbacks = None
    if training_set.has_validation():
        print_and_log("Validation data found")
        validation_data = (training_set.x_validation, training_set.y_validation)
        callbacks = get_callbacks(patience=5)

    history = model.fit(training_set.x_train, training_set.y_train,
                        epochs=epochs,
                        validation_data=validation_data,
                        batch_size=batch_size,
                        callbacks=callbacks)
    print_and_log(f'Final training result ({len(history.history.get("loss"))} epochs): '
                  f'loss: {history.history.get("loss")[-1]}, '
                  f'accuracy: {history.history.get("accuracy")[-1]}')
    if training_set.has_validation():
        print_and_log(f'Validation: val_loss: {history.history.get("val_loss")[-1]}, '
                      f'val_accuracy: {history.history.get("val_accuracy")[-1]}')

    print("Done training!", flush=True)
    #plotter.show_results(history)

    return model, history


def evaluate(model, x_inputs, y_inputs):
    # Load evaluation dataset and model.
    print_and_log("EVALUATION")
    print("Starting evaluation", flush=True)
    batch_size = CONFIG.get("hyper_parameters").get("batch_size")
    scores = model.evaluate(x_inputs, y_inputs, batch_size=batch_size)
    print(f'Done! Evaluation loss and acc: {scores}')
    return scores


# k-fold cross-validation to check how model is performing by selecting different sets to train/validate.
def cross_validate(dataset, num_folds=5):
    # Define the K-fold Cross Validator
    print_and_log("CROSS VALIDATION")
    kfold = KFold(n_splits=num_folds, shuffle=True)

    # K-fold Cross Validation model evaluation
    acc_per_fold = []
    loss_per_fold = []
    fold_no = 1
    for train_index, test_index in kfold.split(dataset.x_combined_bands, dataset.y_output):
        # Generate a print
        print('------------------------------------------------------------------------')
        print_and_log(f'Training for fold {fold_no} ...')

        # Fit data to model
        print_and_log(f'Training fold samples: {dataset.x_combined_bands[train_index].shape[0]}')
        training_set = TrainingSet()
        training_set.x_train = [dataset.x_combined_bands[train_index], dataset.x_angle[train_index]]
        training_set.y_train = dataset.y_output[train_index]
        model, history = train(training_set)

        # Generate generalization metrics
        print_and_log(f'Evaluation fold samples: {dataset.x_combined_bands[test_index].shape[0]}')
        training_set.x_validation = [dataset.x_combined_bands[test_index], dataset.x_angle[test_index]]
        training_set.y_validation = dataset.y_output[test_index]
        scores = evaluate(model, training_set.x_validation, training_set.y_validation)
        print_and_log(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; '
                      f'{model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no = fold_no + 1

    print_and_log("Done with cross-validation!")


# Main code.
def main():
    np.random.seed(555)
    logging.basicConfig(filename='training.log', format='%(asctime)s %(message)s', level=logging.DEBUG)

    # See if we'll use the default or a special config file.
    config_file = CONFIG_FILENAME
    if len(sys.argv) > 1:
        config_file = str(sys.argv[1])
        print('Config file top use: ', config_file)

    # Load config.
    CONFIG.load(config_file)

    print_and_log("--------------------------------------------------------------------")
    print_and_log("Starting trainer session.")

    dataset = augur_dataset.DataSet()
    dataset.load_data(CONFIG.get("dataset"))
    evaluation_input = dataset.get_full_input()
    evaluation_output = dataset.y_output

    # Run steps depending on config.
    if CONFIG.get("training") == "on":
        training_set = split_data(dataset)
        print_and_log(f'Dataset samples {dataset.num_samples}, '
                      f'training samples: {len(training_set.x_train[0])}, '
                      f'validation samples: {len(training_set.x_validation[0])}')

        model, history = train(training_set)
        augur_model.save_model_to_file(model, CONFIG.get("model"))
        evaluation_input = training_set.x_validation
        evaluation_output = training_set.y_validation
    if CONFIG.get("cross_validation") == "on":
        cross_validate(dataset)
    if CONFIG.get("evaluation") == "on":
        model = augur_model.load_model_from_file(CONFIG.get("model"))
        evaluate(model, evaluation_input, evaluation_output)

    print_and_log("Finished trainer session.")
    print_and_log("--------------------------------------------------------------------")


if __name__ == '__main__':
    main()
