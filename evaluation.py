"""
evaluation.py

Forecast error metrics (MAE, RMSE, MAPE, MASE, R2) and the train-set
seasonal-naive denominator used for MASE (Section 3.9).

Requires: Python 3.10, numpy.
Provides: metric functions imported by the model and comparison scripts.
"""

import numpy as np


# MAPE is flagged as unreliable when any |actual| falls below this (MW).
NEAR_ZERO_THRESHOLD = 1.0


def compute_metrics(actual, predicted, mape_floor=0.01,
                    train_series=None, seasonal_period=24):
    """
    Compute the full set of forecast metrics.

    Parameters
    ----------
    actual, predicted : array-like, same length.
        Test-set actuals and predictions.
    mape_floor : float
        Minimum |actual| for inclusion in traditional MAPE. Default 0.01.
    train_series : array-like or None
        Training series used to compute the MASE scaling factor.
        If None, MASE is set to NaN. Must be the original training data
        BEFORE the test set, not the test set itself.
    seasonal_period : int
        Period for seasonal naive baseline in MASE. Default 24 (daily
        seasonality for hourly data). Pass 1 for non-seasonal naive.

    Returns
    -------
    dict with keys:
        MAE, RMSE, R2, NMAE, NRMSE, WAPE, sMAPE, MAPE, MASE,
        min_abs_actual, n_neg_actual, n_used
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) != len(predicted):
        raise ValueError(f"Length mismatch: {len(actual)} vs {len(predicted)}")

    # Drop NaN pairs
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual, predicted = actual[mask], predicted[mask]
    n = len(actual)
    if n == 0:
        return {k: np.nan for k in
                ["MAE", "RMSE", "R2", "NMAE", "NRMSE", "WAPE", "sMAPE",
                 "MAPE", "MASE", "min_abs_actual", "n_neg_actual", "n_used"]}

    err = predicted - actual
    abs_err = np.abs(err)

    # Absolute-scale
    mae = np.mean(abs_err)
    rmse = np.sqrt(np.mean(err ** 2))

    # R^2
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Normalised
    actual_abs_mean = np.mean(np.abs(actual))
    nmae = (mae / actual_abs_mean * 100) if actual_abs_mean > 0 else np.nan
    nrmse = (rmse / actual_abs_mean * 100) if actual_abs_mean > 0 else np.nan

    # WAPE
    sum_abs_actual = np.sum(np.abs(actual))
    wape = (np.sum(abs_err) / sum_abs_actual * 100) if sum_abs_actual > 0 else np.nan

    # sMAPE
    denom = np.abs(actual) + np.abs(predicted)
    smape_mask = denom > 0
    if smape_mask.any():
        smape = np.mean(2 * abs_err[smape_mask] / denom[smape_mask]) * 100
    else:
        smape = np.nan

    # Traditional MAPE (with floor)
    mape_mask = np.abs(actual) > mape_floor
    if mape_mask.any():
        mape = np.mean(np.abs(err[mape_mask] / actual[mape_mask])) * 100
    else:
        mape = np.nan

    # MASE
    mase = compute_mase(actual, predicted, train_series, seasonal_period)

    # Used by format_metrics to decide whether to caveat MAPE.
    min_abs_actual = float(np.min(np.abs(actual)))
    n_neg_actual = int(np.sum(actual < 0))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "NMAE": nmae,
        "NRMSE": nrmse,
        "WAPE": wape,
        "sMAPE": smape,
        "MAPE": mape,
        "MASE": mase,
        "min_abs_actual": min_abs_actual,
        "n_neg_actual": n_neg_actual,
        "n_used": n,
    }


def compute_mase(actual, predicted, train_series, seasonal_period=24):
    """
    Mean Absolute Scaled Error.

    MASE = mean(|test_error|) / mean(|naive_in_sample_error|)

    where naive_in_sample_error[t] = train[t] - train[t - seasonal_period].

    This scales test-set MAE by how hard the series is to predict with a
    seasonal-naive baseline on the training data. Result is dimensionless
    and comparable across datasets of different scales.

    MASE < 1 means the test MAE is below the in-sample seasonal-naive MAE
    on the training data. It is not a comparison with seasonal-naive over
    the test window.

    Parameters
    ----------
    actual, predicted : array-like
        Test-set actuals and predictions.
    train_series : array-like or None
        Training data. If None, returns NaN.
    seasonal_period : int
        Lag for naive baseline. 24 for hourly+daily seasonality, 1 for
        non-seasonal naive (random walk).

    Returns
    -------
    float MASE, or NaN if train_series is unavailable or too short.
    """
    if train_series is None:
        return np.nan

    train = np.asarray(train_series, dtype=float)
    train = train[~np.isnan(train)]
    if len(train) <= seasonal_period:
        return np.nan

    # In-sample MAE of the seasonal-naive forecast on training data
    naive_errors = np.abs(train[seasonal_period:] - train[:-seasonal_period])
    scale = np.mean(naive_errors)

    if scale == 0:
        return np.nan  # degenerate training series

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    test_mae = np.mean(np.abs(predicted[mask] - actual[mask]))

    return test_mae / scale


def format_metrics(m, dataset_hint=None, near_zero_threshold=NEAR_ZERO_THRESHOLD):
    """
    Format metrics dict as a multi-line string.

    MAPE is flagged as unreliable when the actuals include negative values
    or any |actual| below `near_zero_threshold`.

    Parameters
    ----------
    dataset_hint : str or None
        Not used. Accepted so existing calls need not change.
    near_zero_threshold : float
        |actual| below this (in MW) flags MAPE as unreliable.
    """
    lines = [
        f"MAE   = {m['MAE']:.3f} MW",
        f"RMSE  = {m['RMSE']:.3f} MW",
        f"R^2   = {m['R2']:.4f}",
        f"NMAE  = {m['NMAE']:.2f}%   (MAE / mean|actual|)",
        f"NRMSE = {m['NRMSE']:.2f}%   (RMSE / mean|actual|)",
        f"WAPE  = {m['WAPE']:.2f}%   (sum|err| / sum|actual|)",
        f"sMAPE = {m['sMAPE']:.2f}%   (bounded 0-200%)",
    ]

    min_abs = m.get("min_abs_actual", None)
    n_neg = m.get("n_neg_actual", 0)
    mape_unreliable = (n_neg > 0) or (min_abs is not None and min_abs < near_zero_threshold)

    mape_line = f"MAPE  = {m['MAPE']:.2f}%"
    if mape_unreliable:
        if n_neg > 0:
            mape_line += (f"   (unreliable: {n_neg} negative actual(s) in this set; "
                          f"prefer WAPE/MASE)")
        else:
            mape_line += (f"   (unreliable: |actual| approaches zero "
                          f"(min {min_abs:.2f} MW); prefer WAPE/MASE)")
    else:
        note = "reliable: strictly positive load"
        if min_abs is not None:
            note += f", min {min_abs:.2f} MW"
        mape_line += f"   ({note})"
    lines.append(mape_line)

    mase_val = m.get("MASE", np.nan)
    if not np.isnan(mase_val):
        mase_line = f"MASE  = {mase_val:.3f}        (vs seasonal-naive; <1 beats baseline)"
        lines.append(mase_line)
    else:
        lines.append("MASE  = N/A          (training series not passed)")

    return "\n  ".join(lines)