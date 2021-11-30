from scipy import stats


def metric_error(time_interval_id, time_series, ts_predictions):
    """Calculates the Kolmogorov-Smirnov error."""

    x = time_series.get_aggregated(time_interval_id)
    loc = ts_predictions.get_pdf_params(time_interval_id).get("mean")
    scale = ts_predictions.get_pdf_params(time_interval_id).get("std_dev")
    
    ks_stat = stats.kstest([x], "norm", args=(loc, scale)).statistic
    return ks_stat
    