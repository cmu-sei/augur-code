import tensorflow as tf
from tensorflow.keras.metrics import Metric


# Dummy metric for testing Keras custom metrics.
class DummyMetric(Metric):

    def __init__(self, name="dummy", **kwargs):
        super(DummyMetric, self).__init__(name=name, **kwargs)
        self.count = self.add_weight(name="count", initializer="zeros")
        self.diffs = self.add_weight(name="average", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.count.assign_add(1)
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        diff = tf.cast(y_true, "int32") == tf.cast(y_pred, "int32")
        diff = tf.cast(diff, "float32")
        self.diffs.assign_add(diff)
        return

    def result(self):
        return self.count

    def reset_states(self):
        self.diffs.assign(0.0)
        self.count.assign(0.0)
