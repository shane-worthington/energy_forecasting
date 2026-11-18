"""
hybrid_model.py

Fixed 50/50 SARIMA and XGBoost hybrid forecast for a chosen dataset
(Section 3.8.5). Averages the two components' one-step forecasts.

Requires: Python 3.10, pandas, numpy, matplotlib, data_loader, evaluation.
Reads:  results/<dataset>/arima_forecast.csv and xgboost_forecast.csv.
Writes: results/<dataset>/hybrid_forecast.csv, hybrid_results.txt and a plot.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import (
    load_data, get_splits, choose_dataset, get_results_dir, TARGET_COL,
)
from evaluation import compute_metrics, format_metrics

SEASONAL_PERIOD = 24
BLEND_WEIGHT_SARIMA = 0.5   # fixed 50/50; weight on SARIMA, (1-w) on XGBoost


def load_component_forecasts(out_dir):
    """Load and align the SARIMA and XGBoost test forecasts on datetime."""
    sarima_path = out_dir / "arima_forecast.csv"
    xgb_path = out_dir / "xgboost_forecast.csv"
    for p in (sarima_path, xgb_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Component forecast not found: {p}. Run the ARIMA and XGBoost "
                f"scripts for this dataset first."
            )

    sar = pd.read_csv(sarima_path, parse_dates=["datetime"]).set_index("datetime")
    xgb = pd.read_csv(xgb_path, parse_dates=["datetime"]).set_index("datetime")

    if "sarima_pred" not in sar.columns:
        raise ValueError(f"{sarima_path} missing 'sarima_pred' column.")
    if "predicted" not in xgb.columns:
        raise ValueError(f"{xgb_path} missing 'predicted' column.")

    # Inner join on datetime keeps only the test hours both models cover.
    j = sar[["actual", "sarima_pred"]].join(
        xgb[["predicted"]].rename(columns={"predicted": "xgb_pred"}),
        how="inner",
    ).dropna()

    # Actuals come from the SARIMA file. unified_comparison.py checks them
    # against the other forecast files.
    lost_sar = len(sar) - len(j)
    lost_xgb = len(xgb) - len(j)
    if lost_sar or lost_xgb:
        print(f"  Alignment: common rows = {len(j):,} "
              f"(SARIMA dropped {lost_sar}, XGBoost dropped {lost_xgb} non-overlapping).")
    return j


def main():
    print("=" * 65)
    print("Hybrid SARIMA + XGBoost (fixed 50/50 blend)")
    print("=" * 65)

    dataset = choose_dataset()
    out_dir = get_results_dir(dataset)

    j = load_component_forecasts(out_dir)
    actual = j["actual"].values
    sarima_pred = j["sarima_pred"].values
    xgb_pred = j["xgb_pred"].values

    # Fixed equal-weight blend. No test information is used.
    w = BLEND_WEIGHT_SARIMA
    blend = w * sarima_pred + (1.0 - w) * xgb_pred

    # MASE scale from the training split only.
    df = load_data(dataset, include_weather=False)
    train_y, _, _ = get_splits(df, target_only=True, verbose=False)
    train_for_mase = train_y.values

    def metrics_for(pred):
        return compute_metrics(actual, pred, train_series=train_for_mase,
                               seasonal_period=SEASONAL_PERIOD)

    m_sar = metrics_for(sarima_pred)
    m_xgb = metrics_for(xgb_pred)
    m_blend = metrics_for(blend)

    # Test-set optimal weight. Diagnostic only, as it uses test labels.
    ws = np.linspace(0, 1, 101)
    oracle_maes = [np.mean(np.abs((wi * sarima_pred + (1 - wi) * xgb_pred) - actual))
                   for wi in ws]
    oracle_w = float(ws[int(np.argmin(oracle_maes))])
    oracle_mae = float(min(oracle_maes))

    # Correlation between the two components' errors.
    err_sar = sarima_pred - actual
    err_xgb = xgb_pred - actual
    err_corr = float(np.corrcoef(err_sar, err_xgb)[0, 1])

    # Report
    print(f"\nCommon test rows: {len(j):,}")
    print(f"Component error correlation: r = {err_corr:.3f}")
    print("\n" + "=" * 65)
    print(f"Test-set metrics ({dataset}) - one-step-ahead")
    print("=" * 65)
    for label, m in [("SARIMA", m_sar), ("XGBoost", m_xgb),
                     ("Hybrid 50/50", m_blend)]:
        print(f"\n{label}:")
        print(f"  {format_metrics(m, dataset_hint=dataset)}")

    # Did the blend beat both components on MAE?
    beats_both = (m_blend["MAE"] < m_sar["MAE"]) and (m_blend["MAE"] < m_xgb["MAE"])
    outcome = ("B (blend beats BOTH components)" if beats_both
               else "A (blend between components; does not beat the better one)")

    print("\n" + "=" * 65)
    print("Summary (MAE)")
    print("=" * 65)
    print(f"  SARIMA       : {m_sar['MAE']:.4f}")
    print(f"  XGBoost      : {m_xgb['MAE']:.4f}")
    print(f"  Hybrid 50/50 : {m_blend['MAE']:.4f}   <- {'beats both' if beats_both else 'between components'}")
    print(f"  Oracle weight: {oracle_mae:.4f} at w_SARIMA={oracle_w:.2f} (DIAGNOSTIC ONLY, not deployable)")
    print(f"\n  Outcome: {outcome}")

    # Results file
    results_file = out_dir / "hybrid_results.txt"
    with open(results_file, "w") as f:
        f.write(f"Hybrid SARIMA + XGBoost (50/50) results - {dataset}\n")
        f.write("=" * 65 + "\n\n")
        f.write("Fixed 50/50 blend of one-step SARIMA and XGBoost forecasts.\n")
        f.write("Equal weight uses no test/held-out information (leakage-proof).\n\n")
        f.write(f"Common test rows: {len(j):,}\n")
        f.write(f"Component one-step error correlation: r = {err_corr:.3f}\n\n")
        f.write("Component and hybrid metrics:\n\n")
        for label, m in [("SARIMA", m_sar), ("XGBoost", m_xgb),
                         ("Hybrid 50/50", m_blend)]:
            f.write(f"{label}:\n  {format_metrics(m, dataset_hint=dataset)}\n\n")
        f.write("=" * 65 + "\n")
        f.write("Summary (MAE)\n")
        f.write("=" * 65 + "\n")
        f.write(f"  SARIMA       : {m_sar['MAE']:.4f}\n")
        f.write(f"  XGBoost      : {m_xgb['MAE']:.4f}\n")
        f.write(f"  Hybrid 50/50 : {m_blend['MAE']:.4f}\n")
        f.write(f"  Oracle weight: {oracle_mae:.4f} at w_SARIMA={oracle_w:.2f} "
                f"(DIAGNOSTIC ONLY, uses test labels, not deployable)\n\n")
        f.write(f"Outcome: {outcome}\n")
    print(f"\nSaved: {results_file}")

    # Forecast CSV
    forecast_file = out_dir / "hybrid_forecast.csv"
    pd.DataFrame({
        "datetime": j.index,
        "actual": actual,
        "sarima_pred": sarima_pred,
        "xgb_pred": xgb_pred,
        "hybrid_pred": blend,
    }).to_csv(forecast_file, index=False)
    print(f"Saved: {forecast_file}")

    # Plot (first 14 days)
    fig, ax = plt.subplots(figsize=(14, 5))
    n_plot = min(24 * 14, len(j))
    idx = j.index[:n_plot]
    ax.plot(idx, actual[:n_plot], label="Actual", color="black", lw=1.3)
    ax.plot(idx, sarima_pred[:n_plot], label=f"SARIMA (MAE {m_sar['MAE']:.3f})",
            color="#1f77b4", lw=1, alpha=0.7)
    ax.plot(idx, xgb_pred[:n_plot], label=f"XGBoost (MAE {m_xgb['MAE']:.3f})",
            color="#2ca02c", lw=1, alpha=0.7)
    ax.plot(idx, blend[:n_plot], label=f"Hybrid 50/50 (MAE {m_blend['MAE']:.3f})",
            color="#d62728", lw=1.2, alpha=0.9)
    ax.set_ylabel("Load (MW)")
    ax.set_title(f"{dataset.upper()} - SARIMA+XGBoost 50/50 hybrid (first {n_plot//24} days)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plot_file = out_dir / "hybrid_forecast_plot.png"
    plt.savefig(plot_file, dpi=110, bbox_inches="tight")
    print(f"Saved: {plot_file}")


if __name__ == "__main__":
    main()