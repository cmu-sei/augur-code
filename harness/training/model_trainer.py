from sklearn.model_selection import KFold
import tensorflow.keras.callbacks as tfcb

from utils.logging import print_and_log


class ModelTrainer:
    """Has functionalities to train a model"""

    def __init__(self, model_module, config_params):
        self.model_module = model_module
        self.config_params = config_params
        self.evaluation_input = None
        self.evaluation_output = None

    @staticmethod
    def get_callbacks(patience=2):
        """Gets helper callbacks to save checkpoints and allow early stopping when needed."""
        file_path = ".model_weights.hdf5"
        es = tfcb.EarlyStopping('val_loss', patience=patience, mode="min")
        msave = tfcb.ModelCheckpoint(file_path, save_best_only=True)
        return [es, msave]

    def train(self, training_set):
        """Train."""
        print_and_log("TRAINING")

        model = self.model_module.create_model()

        epochs = self.config_params.get("epochs")
        batch_size = self.config_params.get("batch_size")
        print_and_log(f'Starting training with hyper parameters: epochs: {epochs}, batch size: {batch_size}')

        validation_data = None
        callbacks = None
        if training_set.has_validation():
            print_and_log("Validation data found")
            validation_data = (training_set.x_validation, training_set.y_validation)
            callbacks = self.get_callbacks(patience=5)

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

        return model, history

    def evaluate(self, trained_model, evaluation_input=None, evaluation_output=None):
        """Does an evaluation."""
        print_and_log("EVALUATION")
        print("Starting evaluation", flush=True)
        if self.evaluation_input is not None:
            evaluation_input = self.evaluation_input
        if self.evaluation_output is not None:
            evaluation_output = self.evaluation_output
        if evaluation_input is None or evaluation_output is None:
            raise Exception("Evaluation input or output not passed properly to evaluate.")

        batch_size = self.config_params.get("batch_size")
        scores = trained_model.evaluate(evaluation_input, evaluation_output, batch_size=batch_size)
        print(f'Done! Evaluation loss and acc: {scores}')
        return scores

    def cross_validate(self, full_dataset, num_folds=5):
        """k-fold cross-validation to check how model is performing by selecting different sets to train/validate."""

        # Define the K-fold Cross Validator
        print_and_log("CROSS VALIDATION")
        kfold = KFold(n_splits=num_folds, shuffle=True)

        # K-fold Cross Validation model evaluation
        acc_per_fold = []
        loss_per_fold = []
        fold_no = 1
        for train_index, test_index in kfold.split(full_dataset.get_single_input(), full_dataset.get_output()):
            # Generate a print
            print('------------------------------------------------------------------------')
            print_and_log(f'Training for fold {fold_no} ...')

            training_set = self.model_module.get_fold_data(full_dataset, train_index, test_index)

            # Fit data to model
            print_and_log(f'Training fold samples: {training_set.num_train_samples}')
            model, history = self.train(training_set)

            # Generate generalization metrics
            print_and_log(f'Evaluation fold samples: {training_set.num_validation_samples}')
            scores = self.evaluate(model, training_set.x_validation, training_set.y_validation)
            print_and_log(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; '
                          f'{model.metrics_names[1]} of {scores[1] * 100}%')
            acc_per_fold.append(scores[1] * 100)
            loss_per_fold.append(scores[0])

            # Increase fold number
            fold_no = fold_no + 1

        print_and_log("Done with cross-validation!")

    def split_and_train(self, dataset_instance):
        """Splits a dataset and trains the configured model, returning it."""
        training_set = self.model_module.split_data(dataset_instance, self.config_params.get("validation_size"))
        print_and_log(f'Dataset samples {dataset_instance.get_number_of_samples()}, '
                      f'training samples: {len(training_set.x_train[0])}, '
                      f'validation samples: {len(training_set.x_validation[0])}')

        trained_model, history = self.train(training_set)

        # Store evaluation input/outputs as the validation split, in case evaluation is done later.
        self.evaluation_input = training_set.x_validation
        self.evaluation_output = training_set.y_validation
        return trained_model
