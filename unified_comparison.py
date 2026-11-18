"""
unified_comparison.py

Aligns the five models' forecasts on a common test window and computes the
final comparable metrics for each dataset (Section 3.10; Tables 4.1 and 4.2).

Requires: Python 3.10, pandas, numpy, data_loader, evaluation.
Reads:  results/<dataset>/<model>_forecast.csv for all five models.
Writes: results/<dataset>/unified_comparison.csv and unified_comparison.txt.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from data_loader import load_data, get_splits, choose_dataset, get_results_dir
from evaluation import compute_metrics, format_metrics

SEASONAL_PERIOD = 24
BLEND_WEIGHT_SARIMA = 0.5
ACTUAL_TOLERANCE_MW = 1e-6   # max allowed disagreement between files' actuals


def load_and_align(out_dir):
    """Inner-join all component forecast CSVs on datetime.

    Returns a DataFrame indexed by datetime with columns:
      actual, arima_pred, sarima_pred, xgb_pred, lstm_pred
    """
    paths = {
        "arima": out_dir / "arima_forecast.csv",
        "xgb":   out_dir / "xgboost_forecast.csv",
        "lstm":  out_dir / "lstm_forecast.csv",
    }
    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run the {name} script for this dataset first.")

    arima = pd.read_csv(paths["arima"], parse_dates=["datetime"]).set_index("datetime")
    xgb = pd.read_csv(paths["xgb"], parse_dates=["datetime"]).set_index("datetime")
    lstm = pd.read_csv(paths["lstm"], parse_dates=["datetime"]).set_index("datetime")

    j = (arima[["actual", "arima_pred", "sarima_pred"]]
         .join(xgb[["actual", "predicted"]]
               .rename(columns={"actual": "actual_xgb", "predicted": "xgb_pred"}),
               how="inner")
         .join(lstm[["actual", "lstm_pred"]]
               .rename(columns={"actual": "actual_lstm"}),
               how="inner")
         .dropna())

    # Cross-check: all three files must agree on the actuals.
    for other in ["actual_xgb", "actual_lstm"]:
        max_diff = float(np.max(np.abs(j["actual"].values - j[other].values)))
        if max_diff > ACTUAL_TOLERANCE_MW:
            raise ValueError(
                f"Actuals disagree between files ({other}): max |diff| = "
                f"{max_diff:.6f} MW. Check that all scripts were run on the "
                f"same dataset version.")
    j = j.drop(columns=["actual_xgb", "actual_lstm"])

    print(f"  Rows per file: ARIMA {len(arima):,} | XGBoost {len(xgb):,} | "
          f"LSTM {len(lstm):,}")
    print(f"  Common rows after inner join: {len(j):,}")
    print(f"  Common window: {j.index.min()} to {j.index.max()}")
    return j


def gap_diagnostic(df, label):
    """Report gaps longer than one hour in the timestamp index."""
    d = df.index.to_series().diff().dropna()
    gaps = d[d > pd.Timedelta("1h")]
    print(f"\n-- Gap diagnostic ({label}) --")
    print(f"  Rows: {len(df):,} | gaps > 1h: {len(gaps)}")
    if len(gaps):
        print(f"  Largest gap: {gaps.max()}")
        print(f"  Gap sizes (count): ")
        print(gaps.value_counts().sort_index().to_string())
    return len(gaps), (gaps.max() if len(gaps) else pd.Timedelta(0))


def main():
    print("=" * 65)
    print("Unified cross-model comparison (common window, single MASE scale)")
    print("=" * 65)

    dataset = choose_dataset()
    out_dir = get_results_dir(dataset)

    # One MASE scale for every model: the training split (first 70%).
    df = load_data(dataset, include_weather=False)
    train_y, _, _ = get_splits(df, target_only=True, verbose=False)
    train_for_mase = train_y.values
    print(f"\nMASE scale source: TRAIN split only, {len(train_y):,} hours "
          f"({train_y.index.min()} to {train_y.index.max()})")

    # Gap diagnostic on the full loaded series.
    n_gaps, max_gap = gap_diagnostic(df, f"{dataset} full series")

    print("\n-- Aligning forecast files --")
    j = load_and_align(out_dir)

    actual = j["actual"].values
    preds = {
        "ARIMA":        j["arima_pred"].values,
        "SARIMA":       j["sarima_pred"].values,
        "XGBoost":      j["xgb_pred"].values,
        "LSTM":         j["lstm_pred"].values,
    }
    # Recompute hybrid on the common rows (same fixed 50/50 as hybrid_model.py).
    w = BLEND_WEIGHT_SARIMA
    preds["Hybrid 50/50"] = w * preds["SARIMA"] + (1 - w) * preds["XGBoost"]

    results = {}
    for label, p in preds.items():
        results[label] = compute_metrics(
            actual, p,
            train_series=train_for_mase, seasonal_period=SEASONAL_PERIOD)

    # Ranked summary table (by MASE).
    order = sorted(results, key=lambda k: results[k]["MASE"])
    print("\n" + "=" * 65)
    print(f"Unified metrics ({dataset}) - common window, train-only MASE scale")
    print("=" * 65)
    header = f"{'Model':<14}{'MASE':>8}{'MAE':>10}{'RMSE':>10}{'MAPE%':>8}{'R2':>9}"
    print(header)
    print("-" * len(header))
    for k in order:
        m = results[k]
        print(f"{k:<14}{m['MASE']:>8.3f}{m['MAE']:>10.3f}{m['RMSE']:>10.3f}"
              f"{m['MAPE']:>8.2f}{m['R2']:>9.4f}")

    print("\nFull metrics per model:")
    for k in order:
        print(f"\n{k}:")
        print(f"  {format_metrics(results[k], dataset_hint=dataset)}")

    # Results file
    results_file = out_dir / "unified_comparison.txt"
    with open(results_file, "w") as f:
        f.write(f"Unified cross-model comparison - {dataset}\n")
        f.write("=" * 65 + "\n\n")
        f.write("All models evaluated on the COMMON test rows (inner join of\n")
        f.write("all forecast CSVs on datetime) with a SINGLE MASE scale\n")
        f.write("computed from the TRAIN split only (Hyndman & Koehler 2006).\n")
        f.write("Hybrid recomputed as fixed 50/50 blend on the common rows.\n\n")
        f.write(f"Common rows : {len(j):,}\n")
        f.write(f"Window      : {j.index.min()} to {j.index.max()}\n")
        f.write(f"MASE scale  : train only, {len(train_y):,} hours, "
                f"seasonal_period={SEASONAL_PERIOD}\n")
        f.write(f"Index gaps >1h in full series: {n_gaps}"
                + (f" (largest {max_gap})" if n_gaps else "") + "\n\n")
        f.write("Ranked summary (by MASE):\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for k in order:
            m = results[k]
            f.write(f"{k:<14}{m['MASE']:>8.3f}{m['MAE']:>10.3f}{m['RMSE']:>10.3f}"
                    f"{m['MAPE']:>8.2f}{m['R2']:>9.4f}\n")
        f.write("\nFull metrics per model:\n")
        for k in order:
            f.write(f"\n{k}:\n  {format_metrics(results[k], dataset_hint=dataset)}\n")
    print(f"\nSaved: {results_file}")

    # Aligned forecast CSV (one row per common timestamp, all models).
    csv_file = out_dir / "unified_comparison.csv"
    out = j.copy()
    out["hybrid_pred"] = preds["Hybrid 50/50"]
    out.reset_index().to_csv(csv_file, index=False)
    print(f"Saved: {csv_file}")


if __name__ == "__main__":
    main()
