from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import sklearn.model_selection as skm

from training.training_set import TrainingSet

# TODO: Complete this time series model.
# TODO: We may need to add a wrapper function here to actually run the model as well, since we will need to return
# something more than the array of predictions, to include the pdf and pdf params, in a TimeSeries.


def create_model():
    """Model to be used, obtained from sample solution."""

    model = Model()
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


def split_data(dataset, validation_percentage):
    """Split training set into train and validation """
    training_set = TrainingSet()
    return training_set
