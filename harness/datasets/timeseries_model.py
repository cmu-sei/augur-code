from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMAResults

from datasets.timeseries import TimeSeries


def create_fit_model(time_intervals, aggregated_history, interval_unit, params):
    """Creates the model object."""
    model_order = (int(params.get("order_p")), int(params.get("order_q")), int(params.get("order_d")))
    model = ARIMA(aggregated_history, dates=time_intervals, freq=interval_unit, order=model_order)
    fit_model = model.fit()
    return fit_model


def get_prediction_params(fit_model, time_interval_id):
    #TODO: update this to work using dates.
    """Gets the prediction params for the given interval."""
    forecast = fit_model.get_forecast(steps=time_interval_id)

    keep_idx = time_interval_id - 1
    mu = forecast.predicted_mean[keep_idx]
    sigma = forecast.conf_int(alpha=(1-0.68))[keep_idx, 1] - mu
    return {'mean': mu, 'std_dev': sigma}


def predict(fit_model, num_time_intervals):
    """Creates the prediction data, for now only pdf params, based on the fit model."""
    pdf_params = []
    for i in range(0, num_time_intervals):
        pdf_params.append(get_prediction_params(fit_model, i))

    ts_predictions = TimeSeries()
    ts_predictions.set_pdf_params(pdf_params)

    return ts_predictions


def save(ts_trained_model, filename):
    """Save model to file."""
    ts_trained_model.save(filename)


def load(filename):
    """Loads a trained model from file."""
    return ARIMAResults.load(filename)
