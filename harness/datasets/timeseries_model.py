from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMAResults
import pandas as pd

from datasets.timeseries import TimeSeries


def create_fit_model(time_intervals, aggregated_history, interval_unit, params):
    """Creates the model object."""
    # Prepare data as dataframe.
    data = {"Date": time_intervals, "Values": aggregated_history}
    dataframe = pd.DataFrame(data)
    dataframe.set_index("Date", inplace=True)

    # Build and fit model.
    model_order = (int(params.get("order_p")), int(params.get("order_q")), int(params.get("order_d")))
    model = ARIMA(dataframe, freq=interval_unit, order=model_order)
    fit_model = model.fit()
    return fit_model


def get_prediction_params(fit_model, start_interval, end_interval):
    """Gets the prediction params for the given intervals."""
    print(f"Intervals to predict: {start_interval} to {end_interval}")
    forecasts = fit_model.get_prediction(start=start_interval, end=end_interval)

    pdf_params = []
    confidence_level = 0.68
    alpha = 1 - confidence_level
    summary = forecasts.summary_frame(alpha=alpha)
    print(summary)
    for index, row in summary.iterrows():
        mu = row['mean']
        sigma = row['mean_ci_upper'] - mu
        pdf_params.append({'mean': mu, 'std_dev': sigma})

    return pdf_params


def predict(fit_model, time_series):
    """Creates the prediction data, for now only pdf params, based on the fit model."""
    intervals = time_series.get_time_intervals()
    start_interval = intervals[0]
    end_interval = intervals[len(intervals) - 1]

    # Predict the pdf params for each time interval in the series.
    pdf_params = get_prediction_params(fit_model, start_interval, end_interval)

    # Return a time series object with the same intervals plus the pdf params set for each.
    ts_predictions = TimeSeries()
    ts_predictions.set_time_intervals(time_series.get_time_intervals())
    ts_predictions.set_pdf_params(pdf_params)
    return ts_predictions


def save(ts_trained_model, filename):
    """Save model to file."""
    ts_trained_model.save(filename)


def load(filename):
    """Loads a trained model from file."""
    return ARIMAResults.load(filename)
