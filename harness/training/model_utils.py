import tensorflow.keras as keras


# Model saving/loading functions.
def save_model_to_file(model, model_filename):
    model.save(model_filename)


def load_model_from_file(model_filename):
    return keras.models.load_model(model_filename)
