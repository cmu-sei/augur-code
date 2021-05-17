
class TrainingSet(object):
    """Represents a subset of data to train."""
    x_train = []
    y_train = []
    x_validation = None
    y_validation = None
    num_train_samples = 0
    num_validation_samples = 0

    def has_validation(self):
        return self.x_validation is not None and self.y_validation is not None
