import importlib

from analysis.density import DensityEstimator

METRICS_PACKAGE = "extensions.metrics."


def load_metric_module(module_name):
    """Loads a metric module given the name"""
    return importlib.import_module(METRICS_PACKAGE + module_name)


def create_metric(metric_info):
    """Factory method."""
    metric_type = metric_info.get("type")
    if metric_type == "DistanceMetric":
        metric = DistanceMetric()
    elif metric_type == "ErrorMetric":
        metric = ErrorMetric()
    else:
        raise Exception(f"Non-supported metric type: {metric_type}")
    return metric


def check_module_loaded(metric_module):
    """Ensures a metrics module is loaded."""
    if metric_module is not None:
        return True
    else:
        raise Exception("No metric module configured!")


class Metric:
    """Generic Metric class, will load specific metric module as required."""
    metric_module = None
    config_params = None
    time_series = None
    ts_predictions = None

    def load_metric_functions(self, config):
        """Loads a metric module from config."""
        self.metric_module = load_metric_module(config.get("module"))
        self.config_params = config.get("params")

    def initial_setup(self, time_series, ts_predictions):
        """Method to be called once before starting to work with this metric."""
        self.time_series = time_series
        self.ts_predictions = ts_predictions

    def step_setup(self, time_interval_id):
        """Method to be called once in each step/iteration."""
        raise NotImplementedError()

    def calculate_metric(self):
        """Generic method to calculate the value of this metric. Should be overriden by submetrics."""
        raise NotImplementedError()


class ErrorMetric(Metric):
    """Implements an error-based metric that calculates error based on output."""
    time_interval_id = None

    def step_setup(self, time_interval_id):
        """Overriden."""
        self.time_interval_id = time_interval_id

    def calculate_metric(self):
        """Implements an error-based metric."""
        if check_module_loaded(self.metric_module):
            return self.metric_module.metric_error(self.time_interval_id, self.time_series, self.ts_predictions)


class DistanceMetric(Metric):
    """Implements a distance-based metric that can load metric-specific functions from a config."""
    prev_probability_distribution = []  # P
    curr_probability_distribution = []  # Q
    density_estimator = None

    def initial_setup(self, time_series, ts_predictions):
        """Overriden."""
        super().initial_setup(time_series, ts_predictions)
        self.density_estimator = DensityEstimator(self.config_params, self.metric_module)

    def step_setup(self, time_interval_id):
        """Overriden."""
        # Calculate the PD for the data given only some params, and then the PD based only on predictions.
        self.prev_probability_distribution = self.density_estimator.calculate_probability_distribution(self.config_params.get("distribution"),
                                                                                                       self.time_series.get_aggregated(),
                                                                                                       self.ts_predictions.get_pdf_params(time_interval_id))
        self.curr_probability_distribution = self.density_estimator.calculate_probability_distribution(self.config_params.get("distribution"),
                                                                                                       None,
                                                                                                       self.ts_predictions.get_pdf_params(time_interval_id))

    def calculate_metric(self):
        """Calculates the distance defined for the current prob dist and the reference one."""
        if check_module_loaded(self.metric_module):
            return self.metric_module.metric_distance(self.prev_probability_distribution, self.curr_probability_distribution)
