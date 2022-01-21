# Augur: A Step Towards Realistic Drift Detection in Production MLSystems - Code
# Copyright 2022 Carnegie Mellon University.
# 
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
# 
# Released under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
# 
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.
# 
# Carnegie Mellon® is registered in the U.S. Patent and Trademark Office by Carnegie Mellon University.
# 
# This Software includes and/or makes use of the following Third-Party Software subject to its own license:
# 1. Tensorflow (https://github.com/tensorflow/tensorflow/blob/master/LICENSE) Copyright 2014 The Regents of the University of California.
# 2. Pandas (https://github.com/pandas-dev/pandas/blob/main/LICENSE) Copyright 2021 AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team, and open source contributors.
# 3. scikit-learn (https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) Copyright 2021 The scikit-learn developers.
# 4. numpy (https://github.com/numpy/numpy/blob/main/LICENSE.txt) Copyright 2021 NumPy Developers.
# 5. scipy (https://github.com/scipy/scipy/blob/main/LICENSE.txt) Copyright 2021 SciPy Developers.
# 6. statsmodels (https://github.com/statsmodels/statsmodels/blob/main/LICENSE.txt) Copyright 2018 Jonathan E. Taylor, Scipy developers, statsmodels Developers.
# 7. matplotlib (https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE) Copyright 2016 Matplotlib development team.
# 
# DM22-0044

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMAResults
import pandas as pd

from analysis.timeseries import TimeSeries


def create_fit_model(time_intervals, aggregated_history, interval_unit, params):
    """Creates the model object."""
    # Prepare data as dataframe.
    data = {"Date": time_intervals, "Values": aggregated_history}
    dataframe = pd.DataFrame(data)
    dataframe.set_index("Date", inplace=True)

    # Rebuild index to avoid warning and issue with frequency.
    tidx = pd.DatetimeIndex(dataframe.index.values, freq=interval_unit)
    dataframe.set_index(tidx, inplace=True)
    print(dataframe)

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
