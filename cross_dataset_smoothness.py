"""
cross_dataset_smoothness.py

Cross-dataset load smoothness metrics for Section 4.5 (coefficient of
variation, mean hour-to-hour change, and percentile and peak-to-trough
ratios), computed on the full hourly load series of each dataset.

Requires: Python 3.10, pandas, numpy, data_loader.
Reads:  the load series via data_loader (load only, missing hours dropped).
Writes: results/smoothness.txt.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import load_data, TARGET_COL

# Datasets in output order: national first, then the substation.
DATASETS = [
    ("spain", "National (Spain)"),
    ("ee", "Western Sydney substation"),
]


def smoothness_metrics(series):
    """Return the four smoothness metrics for a single load series."""
    s = series.dropna().astype(float)
    mean = s.mean()
    return {
        "cv": s.std() / mean,
        "h2h_pct": s.diff().abs().mean() / mean * 100.0,
        "p95_p5": s.quantile(0.95) / s.quantile(0.05),
        "max_min": s.max() / s.min(),
        "n": len(s),
    }


def main():
    results = {}
    for key, _ in DATASETS:
        df = load_data(key, include_weather=False, drop_missing=True)
        results[key] = smoothness_metrics(df[TARGET_COL])

    # Build the comparison table.
    lines = []
    lines.append("Cross-dataset smoothness (full hourly load series)")
    lines.append("")
    header = f"{'Metric':<34}" + "".join(f"{label:>26}" for _, label in DATASETS)
    lines.append(header)
    lines.append("-" * len(header))

    rows = [
        ("Observations (hours)", "n", "{:,.0f}"),
        ("Coefficient of variation", "cv", "{:.3f}"),
        ("Mean hour-to-hour change (% mean)", "h2h_pct", "{:.2f}"),
        ("95th / 5th percentile ratio", "p95_p5", "{:.2f}"),
        ("Maximum / minimum ratio", "max_min", "{:.2f}"),
    ]
    for label, field, fmt in rows:
        cells = "".join(f"{fmt.format(results[k][field]):>26}" for k, _ in DATASETS)
        lines.append(f"{label:<34}{cells}")

    # Substation / national ratios for the two headline measures.
    lines.append("")
    lines.append("Substation / national ratio:")
    for label, field in [("Coefficient of variation", "cv"),
                         ("Mean hour-to-hour change", "h2h_pct")]:
        ratio = results["ee"][field] / results["spain"][field]
        lines.append(f"  {label:<32}{ratio:>6.2f}x")

    report = "\n".join(lines)
    print("\n" + report + "\n")

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoothness.txt").write_text(report + "\n", encoding="utf-8")
    print(f"Written: {(out_dir / 'smoothness.txt').resolve()}")


if __name__ == "__main__":
    main()
