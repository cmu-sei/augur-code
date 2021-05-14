import tensorflow.keras as keras


# Model saving/loading functions.

def save_model_to_file(model, model_filename):
    model.save(model_filename)


def load_model_from_file(model_filename):
    return keras.models.load_model(model_filename)


def add_metric(model, metric):
    model.compile(optimizer=model.optimizer,
                  loss=model.loss,
                  metrics=model.metrics + [metric])


def add_metrics(model, metrics):
    model.compile(optimizer=model.optimizer,
                  loss=model.loss,
                  metrics=model.metrics + metrics)
