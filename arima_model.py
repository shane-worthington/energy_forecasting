"""
arima_model.py

One-step-ahead ARIMA and SARIMA forecasts for a chosen dataset
(Sections 3.8.1 and 3.8.2).

Requires: Python 3.10, pandas, numpy, statsmodels, pmdarima, matplotlib,
          data_loader, evaluation.
Reads:  the chosen dataset via data_loader.
Writes: results/<dataset>/arima_forecast.csv, arima_results.txt and a plot.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_data, get_splits, choose_dataset, get_results_dir, TARGET_COL
from evaluation import compute_metrics, format_metrics

try:
    from pmdarima import auto_arima
except ImportError as e:
    raise ImportError("pmdarima not installed. Run: pip install pmdarima") from e

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller


# Configuration

SEASONAL_PERIOD = 24
N_TEST_STEPS = None   # None = forecast full test; set int for a quick dry-run

# Parameters are estimated once on the fit history, then held fixed.
FIT_MAXITER = 200
ENFORCE_STATIONARITY = True
ENFORCE_INVERTIBILITY = True

# Per-point sanity bound: fall back to seasonal-naive if a one-step forecast
# is non-finite or exceeds SANITY_MULT * training-range max abs.
SANITY_MULT = 5.0

# Orders selected by auto_arima for each dataset. A dataset listed here
# skips the auto_arima search.
FIXED_SPECS = {
    "spain": {
        "ns_order":   (2, 1, 0),
        "s_order":    (2, 1, 0),
        "s_seasonal": (2, 0, 1, SEASONAL_PERIOD),
    },
    "ee": {
        "ns_order":   (4, 1, 4),
        "s_order":    (2, 1, 0),
        "s_seasonal": (2, 0, 1, SEASONAL_PERIOD),
    },
}


# Stationarity check

def check_stationarity(series, name="series"):
    result = adfuller(series.dropna(), autolag="AIC")
    stat, pval = result[0], result[1]
    print(f"  ADF on {name}: statistic={stat:.3f}, p-value={pval:.4f} "
          f"-> {'stationary' if pval < 0.05 else 'non-stationary'}")
    return pval < 0.05


# Order selection

# auto_arima searches on the most recent MAX_SEARCH_HOURS only, to keep the
# seasonal search tractable. The final fit uses the full history.
MAX_SEARCH_HOURS = 12000

def find_arima_specs(train_y, dataset_name=""):
    """Run auto_arima twice (non-seasonal and seasonal). Returns orders."""
    if len(train_y) > MAX_SEARCH_HOURS:
        search_y = train_y.iloc[-MAX_SEARCH_HOURS:]
        print(f"\n  Note: using last {MAX_SEARCH_HOURS:,} hours of training "
              f"data for spec search (of {len(train_y):,} total).")
        print(f"  Full training history is used for the final fit.")
    else:
        search_y = train_y

    print("\n-- Fitting non-seasonal auto_arima --")
    model_ns = auto_arima(
        search_y, seasonal=False,
        d=None, start_p=0, start_q=0, max_p=5, max_q=5,
        information_criterion="aic",
        stepwise=True, suppress_warnings=True, error_action="ignore", trace=False,
    )
    print(f"  Selected: ARIMA{model_ns.order}, AIC: {model_ns.aic():.2f}")

    print("\n-- Fitting seasonal auto_arima (m=24) --")
    model_s = auto_arima(
        search_y, seasonal=True, m=SEASONAL_PERIOD,
        d=None, D=None,
        start_p=0, start_q=0, start_P=0, start_Q=0,
        max_p=3, max_q=3, max_P=1, max_Q=1,
        information_criterion="aic",
        stepwise=True, suppress_warnings=True, error_action="ignore", trace=False,
    )
    print(f"  Selected: SARIMA{model_s.order}{model_s.seasonal_order}[{SEASONAL_PERIOD}], "
          f"AIC: {model_s.aic():.2f}")

    if dataset_name:
        print("\n" + "-"*65)
        print("To skip auto_arima on subsequent runs, paste this into FIXED_SPECS:")
        print("-"*65)
        print(f'    "{dataset_name}": {{')
        print(f'        "ns_order":   {tuple(model_ns.order)},')
        print(f'        "s_order":    {tuple(model_s.order)},')
        print(f'        "s_seasonal": {tuple(model_s.seasonal_order)},')
        print(f'    }},')
        print("-"*65)

    return model_ns.order, model_s.order, model_s.seasonal_order


# One-step-ahead forecasting

def one_step_forecast(history, order, seasonal_order, test_y,
                      n_steps=None, label="ARIMA"):
    """
    One-step-ahead forecast over the test set.

    Parameters are estimated once on the fit history. The same model is then
    applied to the combined train+test series with those parameters held
    fixed (.filter(), no re-estimation). get_prediction(dynamic=False) gives
    the prediction for each test hour t from actual observations up to t-1.
    This is equivalent to appending each actual and updating the filter one
    step at a time, but runs in a single pass.

    Any non-finite or out-of-range prediction is replaced with the
    seasonal-naive value (the actual 24 hours earlier).

    Returns an array aligned to test_y.index[:n_steps].
    """
    print(f"\n-- One-step-ahead forecast with {label} --")
    n_test = len(test_y)
    if n_steps is not None:
        n_test = min(n_test, n_steps)

    hist_max_abs = float(np.max(np.abs(history.values)))
    sanity_bound = SANITY_MULT * hist_max_abs

    # Plain arrays are used because the indices have no fixed frequency after
    # the time zone conversion. Predictions are re-aligned to dates in main().
    hist_vals = np.asarray(history.values, dtype=float)
    test_vals = np.asarray(test_y.values, dtype=float)[:n_test]

    # Estimate parameters on the fit history.
    print(f"  Fitting {label} once on {len(hist_vals):,} hours "
          f"(maxiter={FIT_MAXITER})...")
    res_train = SARIMAX(
        hist_vals,
        order=order, seasonal_order=seasonal_order,
        enforce_stationarity=ENFORCE_STATIONARITY,
        enforce_invertibility=ENFORCE_INVERTIBILITY,
    ).fit(disp=False, maxiter=FIT_MAXITER)
    print(f"  Fit complete. Applying frozen params over {n_test:,} "
          f"one-step points...")

    # Apply the fixed parameters to train+test and take one-step
    # predictions over the test span.
    full_vals = np.concatenate([hist_vals, test_vals])
    full_model = SARIMAX(
        full_vals,
        order=order, seasonal_order=seasonal_order,
        enforce_stationarity=ENFORCE_STATIONARITY,
        enforce_invertibility=ENFORCE_INVERTIBILITY,
    )
    full_res = full_model.filter(res_train.params)

    start = len(hist_vals)
    end = len(full_vals) - 1
    pred = full_res.get_prediction(start=start, end=end, dynamic=False)
    predictions = np.asarray(pred.predicted_mean, dtype=float)

    # Seasonal-naive fallback. The first 24 test hours draw on the end of
    # the fit history.
    n_fallback = 0
    bad = ~np.isfinite(predictions) | (np.abs(predictions) > sanity_bound)
    if bad.any():
        for i in np.where(bad)[0]:
            naive_idx = start + i - SEASONAL_PERIOD
            predictions[i] = full_vals[naive_idx] if naive_idx >= 0 else full_vals[start + i - 1]
            n_fallback += 1

    if n_fallback:
        pct = 100 * n_fallback / n_test
        print(f"  Note: {n_fallback}/{n_test} points ({pct:.2f}%) used the "
              f"seasonal-naive fallback (non-finite or out-of-range).")
    else:
        print(f"  Done. No fallbacks needed.")

    return predictions


# Main

def main():
    print("="*65)
    print("ARIMA / SARIMA pipeline (one-step-ahead)")
    print("="*65)

    dataset = choose_dataset()
    out_dir = get_results_dir(dataset)

    # ARIMA is univariate; weather not needed.
    df = load_data(dataset, include_weather=False)
    train_y, val_y, test_y = get_splits(df, target_only=True)

    fit_history = pd.concat([train_y, val_y])
    print(f"\nFit history: {len(fit_history):,} hours | "
          f"{fit_history.index.min()} to {fit_history.index.max()}")
    print(f"Test set   : {len(test_y):,} hours | "
          f"{test_y.index.min()} to {test_y.index.max()}")

    print("\n-- Stationarity check --")
    check_stationarity(fit_history, "raw load")
    check_stationarity(fit_history.diff().dropna(), "1st-diff load")

    if dataset in FIXED_SPECS:
        spec = FIXED_SPECS[dataset]
        ns_order = spec["ns_order"]
        s_order = spec["s_order"]
        s_seasonal = spec["s_seasonal"]
        print(f"\n-- Using fixed ARIMA specs for {dataset} (skipping auto_arima) --")
        print(f"  Non-seasonal: ARIMA{ns_order}")
        print(f"  Seasonal:     SARIMA{s_order}{s_seasonal}")
        print(f"  (To re-derive, remove '{dataset}' from FIXED_SPECS.)")
    else:
        ns_order, s_order, s_seasonal = find_arima_specs(fit_history, dataset_name=dataset)

    pred_ns = one_step_forecast(fit_history, ns_order, (0, 0, 0, 0),
                                test_y, n_steps=N_TEST_STEPS, label="ARIMA")
    pred_s = one_step_forecast(fit_history, s_order, s_seasonal,
                               test_y, n_steps=N_TEST_STEPS, label="SARIMA")

    n_done = len(pred_ns)
    test_eval = test_y.iloc[:n_done]

    print("\n" + "="*65)
    print(f"Test-set metrics ({dataset}) - one-step-ahead")
    print("="*65)
    train_for_mase = fit_history.values
    metrics_ns = compute_metrics(test_eval.values, pred_ns,
                                 train_series=train_for_mase, seasonal_period=SEASONAL_PERIOD)
    metrics_s = compute_metrics(test_eval.values, pred_s,
                                train_series=train_for_mase, seasonal_period=SEASONAL_PERIOD)

    print(f"\nARIMA{ns_order}:")
    print(f"  {format_metrics(metrics_ns, dataset_hint=dataset)}")
    print(f"\nSARIMA{s_order}{s_seasonal}[{SEASONAL_PERIOD}]:")
    print(f"  {format_metrics(metrics_s, dataset_hint=dataset)}")

    results_file = out_dir / "arima_results.txt"
    with open(results_file, "w") as f:
        f.write(f"ARIMA / SARIMA results - {dataset}\n")
        f.write("="*65 + "\n\n")
        f.write("Strategy: one-step-ahead (fit once on training history,\n")
        f.write("frozen parameters, append actuals with Kalman update only).\n")
        f.write("Matches the one-step convention of the other models.\n\n")
        f.write(f"Fit history: {len(fit_history):,} hours | "
                f"{fit_history.index.min()} to {fit_history.index.max()}\n")
        f.write(f"Test set   : {len(test_eval):,} hours | "
                f"{test_eval.index.min()} to {test_eval.index.max()}\n\n")
        f.write(f"Non-seasonal: ARIMA{ns_order}\n")
        f.write(f"  {format_metrics(metrics_ns, dataset_hint=dataset)}\n\n")
        f.write(f"Seasonal: SARIMA{s_order}{s_seasonal}[{SEASONAL_PERIOD}]\n")
        f.write(f"  {format_metrics(metrics_s, dataset_hint=dataset)}\n")
    print(f"\nSaved: {results_file}")

    forecast_file = out_dir / "arima_forecast.csv"
    pd.DataFrame({
        "datetime": test_eval.index,
        "actual": test_eval.values,
        "arima_pred": pred_ns,
        "sarima_pred": pred_s,
    }).to_csv(forecast_file, index=False)
    print(f"Saved: {forecast_file}")

    fig, ax = plt.subplots(figsize=(14, 5))
    n_plot = min(24 * 14, len(test_eval))
    ax.plot(test_eval.index[:n_plot], test_eval.values[:n_plot],
            label="Actual", color="black", lw=1.2)
    ax.plot(test_eval.index[:n_plot], pred_ns[:n_plot],
            label=f"ARIMA{ns_order}", color="#d62728", lw=1, alpha=0.8)
    ax.plot(test_eval.index[:n_plot], pred_s[:n_plot],
            label=f"SARIMA{s_order}{s_seasonal}[{SEASONAL_PERIOD}]",
            color="#1f77b4", lw=1, alpha=0.8)
    ax.set_ylabel("Load (MW)")
    ax.set_title(f"{dataset.upper()} - ARIMA one-step-ahead forecast (first {n_plot//24} days)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plot_file = out_dir / "arima_forecast_plot.png"
    plt.savefig(plot_file, dpi=110, bbox_inches="tight")
    print(f"Saved: {plot_file}")


if __name__ == "__main__":
    main()