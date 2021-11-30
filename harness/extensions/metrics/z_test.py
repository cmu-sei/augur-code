
def metric_error(time_interval_id, time_series, ts_predictions):
    """Calculates the STEPD error."""
    x1 = time_series.get_aggregated(time_interval_id)
    x2 = ts_predictions.get_pdf_params(time_interval_id).get("mean")
    sigma = ts_predictions.get_pdf_params(time_interval_id).get("std_dev")
    t = (x1 - x2) / sigma
    return t
